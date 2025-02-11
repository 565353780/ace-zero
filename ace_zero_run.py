import os
import time
import shutil
import numpy as np
import ace_zero_util as zutil
from joblib import Parallel, delayed

import dataset_io

from ace_zero.Config.config import getArgs


if __name__ == '__main__':
    opt = getArgs()

    # create output directory
    os.makedirs(opt.results_folder, exist_ok=True)

    print(f"Starting reconstruction of files matching {opt.rgb_files}.")

    print(f"Downloading ZoeDepth model from the main process.")
    model = dataset_io.get_depth_model()
    del model
    print(f"Depth estimation model ready to use.")

    reconstruction_start_time = time.time()

    if opt.seed_network is not None:
        print(f"Using pre-trained network as seed: {opt.seed_network}")
        iteration_id = opt.seed_network.stem
    else:
        # use individual images as seeds, try multiple and choose the one that registers the most images
        np.random.seed(opt.random_seed)
        seeds = np.random.uniform(size=opt.try_seeds)
        print(f"Trying seeds: {seeds}")

        # process seeds in parallel
        arg_instances = []
        for seed_idx, seed in enumerate(seeds):
            # show progress only for the first seed or if we are not using parallel workers
            verbose = (seed_idx == 0) or (opt.seed_parallel_workers == 1)
            arg_instances.append((seed_idx, seed, opt.rgb_files, opt.results_folder, opt, verbose, False, False))

        if opt.seed_parallel_workers != 1:
            print(f"Processing {len(arg_instances)} seeds in parallel.")

        # as we process initial seeds, keep track of their registration rates
        seed_reg_rates = Parallel(n_jobs=opt.seed_parallel_workers, verbose=1)(
            map(delayed(zutil.map_seed), arg_instances))

        for seed_idx, seed in enumerate(seeds):
            print(f"Seed {seed_idx}: {seed} -> {seed_reg_rates[seed_idx] * 100:.1f}%")

        # select the best seed
        best_seed = np.argmax(seed_reg_rates)
        iteration_id = zutil.get_seed_id(best_seed)

        print(f"Selected best seed {iteration_id} "
                     f"with registration rate: {seed_reg_rates[best_seed] * 100:.1f}%")

        # if a visualisation is requested, we need to re-map the best seed with visualisation enabled
        if opt.render_visualization:
            print(f"Re-mapping best seed {iteration_id} with visualisation enabled.")
            zutil.map_seed((best_seed, seeds[best_seed], opt.rgb_files, opt.results_folder, opt, True, True, True))

    print(f"Registering all images to best seed {iteration_id}.")

    # Register all images to the best seed. Also render visualisation if requested.
    # In some cases, this is redundant - when the dataset is small and the seed scoring already registered all images
    # AND no visualisation was requested. However, for small datasets, this is fast anyway.
    reg_cmd = [
        zutil.REGISTER_EXE,
        opt.rgb_files,
        opt.results_folder / f"{iteration_id}.pt",
        "--render_visualization", opt.render_visualization,
        "--render_target_path", zutil.get_render_path(opt.results_folder),
        "--render_marker_size", opt.render_marker_size,
        "--render_flipped_portrait", opt.render_flipped_portrait,
        "--session", f"{iteration_id}",
        "--confidence_threshold", opt.registration_confidence,
        "--use_external_focal_length", opt.use_external_focal_length,
        "--hypotheses", opt.ransac_iterations,
        "--threshold", opt.ransac_threshold,
        "--image_resolution", opt.image_resolution,
        "--num_data_workers", opt.num_data_workers,
        "--hypotheses_max_tries", 16
    ]
    zutil.run_cmd(reg_cmd)

    scheduled_to_stop_early = False
    prev_iteration_id = iteration_id

    # check the number of registered mapping images
    max_registration_rate = zutil.get_registration_rates(
        pose_file=opt.results_folder / f"poses_{iteration_id}.txt",
        thresholds=[opt.registration_confidence])[0]
    print(f"Best seed successfully registered {max_registration_rate * 100:.1f}% of mapping images.")

    # iterate mapping and registration starting from the best seed iteration
    for iteration in range(1, opt.iterations_max):

        iteration_id = f"iteration{iteration}"

        if scheduled_to_stop_early and opt.final_refit:
            # get full refitting mapping call
            mapping_cmd = zutil.get_refit_mapping_cmd(opt.rgb_files, iteration_id, opt.results_folder, opt)
        else:
            # get base mapping call
            mapping_cmd = zutil.get_base_mapping_cmd(opt.rgb_files, iteration_id, opt.results_folder, opt)

        # setting parameters for mapping after initial seed
        mapping_cmd += [
            "--render_visualization", opt.render_visualization,
            "--use_ace_pose_file", f"{opt.results_folder}/poses_{prev_iteration_id}.txt",
            "--pose_refinement", opt.refinement,
            "--use_existing_vis_buffer", f"{prev_iteration_id}_register.pkl",
            "--refine_calibration", opt.refine_calibration,
            "--num_data_workers", opt.num_data_workers,
        ]

        # load previous network weights starting from iteration 2, or if we started from a seed network
        if (opt.warmstart and iteration > 1) or (opt.warmstart and opt.seed_network is not None):
            # skip warmstart in final refit iteration
            if not (opt.final_refit and scheduled_to_stop_early):
                mapping_cmd += ["--load_weights", f"{opt.results_folder}/{prev_iteration_id}.pt"]

        zutil.run_cmd(mapping_cmd)

        # register all images to the current map
        print(f"Registering all images to map {iteration_id}.")

        reg_cmd = [
            zutil.REGISTER_EXE,
            opt.rgb_files,
            opt.results_folder / f"{iteration_id}.pt",
            "--render_visualization", opt.render_visualization,
            "--render_target_path", zutil.get_render_path(opt.results_folder),
            "--render_marker_size", opt.render_marker_size,
            "--session", iteration_id,
            "--confidence_threshold", opt.registration_confidence,
            "--render_flipped_portrait", opt.render_flipped_portrait,
            "--image_resolution", opt.image_resolution,
            "--hypotheses", opt.ransac_iterations,
            "--threshold", opt.ransac_threshold,
            "--num_data_workers", opt.num_data_workers,
            "--hypotheses_max_tries", 16
        ]

        # Get current focal length estimate from the pose file of the previous mapping iteration
        _, _, focal_lengths = dataset_io.load_dataset_ace(
            pose_file=opt.results_folder / f"poses_{iteration_id}_preliminary.txt",
            confidence_threshold=opt.registration_confidence)

        # We support a single focal length.
        assert np.allclose(focal_lengths, focal_lengths[0])

        print("Passing previous focal length estimate to registration.")
        reg_cmd += ["--use_external_focal_length", focal_lengths[0]]

        zutil.run_cmd(reg_cmd)

        # check the number of registered mapping images
        registration_rate = zutil.get_registration_rates(
            pose_file=opt.results_folder / f"poses_{iteration_id}.txt",
            thresholds=[opt.registration_confidence])[0]

        print(f"Successfully registered {registration_rate*100:.1f}% of mapping images.")

        prev_iteration_id = iteration_id

        if scheduled_to_stop_early:
            # we are in the final refinement iteration and stop here
            break

        # check stopping criteria
        if (registration_rate >= opt.registration_threshold) or ((registration_rate-max_registration_rate) < opt.relative_registration_threshold):
            if opt.final_refine:
                # stopping criteria have been met, but we want to do one more round of mapping
                print(f"Stopping training loop in next iteration. Enough mapping images registered. "
                             f"(Threshold={opt.registration_threshold * 100:.1f}%")
                scheduled_to_stop_early = True
            else:
                # stopping criteria have been met, and we do not want to do one more round of mapping
                print(f"Stopping training loop. Enough mapping images registered. "
                             f"(Threshold={opt.registration_threshold * 100:.1f}%")
                break

        # stop in any case if we are approaching the maximum number of iterations
        if iteration >= (opt.iterations_max - 2):
            scheduled_to_stop_early = True

        max_registration_rate = max(registration_rate, max_registration_rate)

    if opt.render_visualization:
        print("Rendering final sweep.")

        zutil.run_cmd(["./render_final_sweep.py",
                       zutil.get_render_path(opt.results_folder),
                       "--render_marker_size", opt.render_marker_size
                       ])

        print("Converting to video.")

        # get ffmpeg path
        ffmpeg_path = shutil.which("ffmpeg")

        # run ffmpeg to convert the rendered images to a video
        zutil.run_cmd([ffmpeg_path,
                       "-y",
                       "-framerate", 30,
                       "-pattern_type", "glob",
                       "-i", f"{zutil.get_render_path(opt.results_folder)}/*.png",
                       "-c:v", "libx264",
                       "-pix_fmt", "yuv420p",
                       opt.results_folder / "reconstruction.mp4"
                       ])

    reconstruction_end_time = time.time()
    reconstruction_time = reconstruction_end_time - reconstruction_start_time
    print(f"Reconstructed in {reconstruction_time/60:.1f} Minutes.")

    # check the number of registered mapping images
    registration_rates = zutil.get_registration_rates(
        pose_file=opt.results_folder / f"poses_{iteration_id}.txt",
        thresholds=[500, 1000, 2000, 4000])

    # copy pose estimates of the final iteration to output file
    final_pose_file = opt.results_folder / f"poses_{iteration_id}.txt"
    shutil.copy(final_pose_file, final_pose_file.parent / f"poses_final.txt")

    # export point cloud if requested
    if opt.export_point_cloud:
        print("Exporting point cloud.")

        if not opt.dense_point_cloud and opt.render_visualization:
            vis_buffer_file = zutil.get_render_path(opt.results_folder) / f"{iteration_id}_mapping.pkl",
            print(f"Exporting point cloud from visualisation buffer file: {vis_buffer_file}")

            zutil.run_cmd(["./export_point_cloud.py",
                           opt.results_folder / "pc_final.ply",
                           "--visualization_buffer", vis_buffer_file,
                           "--convention", "opencv",
                           ])
        else:
            print(f"Exporting point cloud from last network and pose file.")

            zutil.run_cmd(["./export_point_cloud.py",
                           opt.results_folder / "pc_final.ply",
                           "--network", opt.results_folder / f"{iteration_id}.pt",
                           "--pose_file", opt.results_folder / f"poses_final.txt",
                           "--convention", "opencv",
                           "--dense_point_cloud", opt.dense_point_cloud,
                           ])

    stats_report = "Time (min) | Iterations | Reg. Rate @500 | @1000 | @2000 | @4000\n"
    stats_report += f"{reconstruction_time / 60:.1f} " \
                    f"{iteration} " \
                    f"{registration_rates[0] * 100:.1f}% " \
                    f"{registration_rates[1] * 100:.1f}% " \
                    f"{registration_rates[2] * 100:.1f}% " \
                    f"{registration_rates[3] * 100:.1f}%\n"

    print(stats_report)
