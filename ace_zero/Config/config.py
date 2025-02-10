import argparse
from pathlib import Path
from distutils.util import strtobool


def _strtobool(x):
    return bool(strtobool(x))

def getArgs():
    parser = argparse.ArgumentParser(
        description='Run ACE0 for a dataset or a scene.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('rgb_files', type=str, help="Glob pattern for RGB files, e.g. 'datasets/scene/*.jpg'")

    parser.add_argument('results_folder', type=Path, help='path to output folder for result files')

    parser.add_argument('--depth_files', type=str, default=None,
                        help="Which depth to use for the seed image, Glob pattern. "
                             "Correspondence to rgb files via alphateical ordering. "
                             "None: estimate depth using ZoeDepth")

    # === Main reconstruction loop =====================================================================================

    parser.add_argument('--iterations_max', type=int, default=100,
                        help="Maximum number of ACE0 iterations, ie mapping and relocalisation rounds.")
    parser.add_argument('--registration_threshold', type=float, default=0.99,
                        help="Stop reconstruction when this ratio of images has been registered.")
    parser.add_argument('--relative_registration_threshold', type=float, default=0.01,
                        help="Stop reconstruction when less percent of images was registered wrt the last iteration.")
    parser.add_argument('--final_refine', type=_strtobool, default=True,
                        help="One more round of mapping when the stopping criteria have been met.")
    parser.add_argument('--final_refit', type=_strtobool, default=True,
                        help="Refit new (uninitialised) network in last iteration without early stopping")
    parser.add_argument('--final_refit_posewait', type=int, default=5000,
                        help="Fix poses for the first n training iterations of the final refit.")
    parser.add_argument('--refit_iterations', type=int, default=25000,
                        help='Number of training iterations for the final refit.')
    parser.add_argument('--registration_confidence', type=int, default=500,
                        help="Consider an image registered if it has this many inlier scene coordinates.")

    parser.add_argument('--try_seeds', type=int, default=5,
                        help="Number of random images to try when starting the reconstruction.")
    parser.add_argument('--seed_parallel_workers', type=int, default=3,
                        help="Number of parallel workers for seed mapping. "
                             "ZoeDepth might be a limiting factor in terms of GPU memory. "
                             "-1 -> all available cores, 1 -> no parallelism.")
    parser.add_argument('--seed_iterations', type=int, default=10000,
                        help="Maximum number of ACE training iterations for seed images.")

    parser.add_argument('--seed_network', type=Path, default=None,
                        help="Path to a pre-trained network to start the reconstruction.")

    parser.add_argument('--warmstart', type=_strtobool, default=True,
                        help="For each ACE0 mapping round, load the ACE weights of the last iteration.")

    parser.add_argument('--export_point_cloud', type=_strtobool, default=False,
                        help="Export the ACE0 point cloud after reconstruction, "
                             "for visualisation or to initialise splats")

    parser.add_argument('--dense_point_cloud', type=_strtobool, default=False,
                        help='when exporting a point cloud, do not filter points based on reprojection error, '
                             'bad for visualisation but good to initialise splats')

    # === Pose refinement ==================================================================================================

    parser.add_argument('--refinement', type=str, default="mlp", choices=['mlp', 'none', 'naive'],
                        help="How to refine poses. MLP: refinement network. Naive: Backprop to poses.")
    parser.add_argument('--refinement_ortho', type=str, default="gram-schmidt", choices=['gram-schmidt', 'procrustes'],
                        help="How to orthonormalise rotations when refining poses.")
    parser.add_argument('--pose_refinement_wait', type=int, default=0,
                        help="Keep poses frozen for the first n training iterations of ACE.")
    parser.add_argument('--pose_refinement_lr', type=float, default=0.001,
                        help="Learning rate for pose refinement.")

    # === Calibration refinement ===========================================================================================

    parser.add_argument('--refine_calibration', type=_strtobool, default=True,
                        help="Optimize focal length during mapping.")
    parser.add_argument('--use_external_focal_length', type=float, default=-1,
                        help="Provide the focal length of images. Can be refined. "
                             "-1: Use 70% of image diagonal as guess.")

    # === ACE Early Stopping ===============================================================================================

    parser.add_argument('--learning_rate_schedule', type=str, default="1cyclepoly",
                        choices=["circle", "constant", "1cyclepoly"],
                        help='circle: move from min to max to min, constant: stay at min, '
                             '1cyclepoly: linear approximation of 1cycle to support early stopping')
    parser.add_argument('--learning_rate_max', type=float, default=0.003, help="max learning rate of the lr schedule")
    parser.add_argument('--cooldown_iterations', type=int, default=5000,
                        help="train to min learning rate for this many iterations after early stopping criterion has been met")
    parser.add_argument('--cooldown_threshold', type=float, default=0.7,
                        help="Start cooldown after this percent of batch pixels are below the inlier reprojection error.")

    # === General ACE parameters ===========================================================================================

    parser.add_argument('--image_resolution', type=int, default=480,
                        help='base image resolution')
    parser.add_argument('--num_head_blocks', type=int, default=1,
                        help='depth of the regression head, defines the map size')
    parser.add_argument('--max_dataset_passes', type=int, default=10,
                        help='max number of repetition of mapping images (with different augmentations)')
    parser.add_argument('--repro_loss_type', type=str, default="tanh",
                        choices=["l1", "l1+sqrt", "l1+log", "tanh", "dyntanh"],
                        help='Loss function on the reprojection error. Dyn varies the soft clamping threshold')
    parser.add_argument('--repro_loss_hard_clamp', type=int, default=1000,
                        help='hard clamping threshold for the reprojection losses')
    parser.add_argument('--repro_loss_soft_clamp', type=int, default=50,
                        help='soft clamping threshold for the reprojection losses')
    parser.add_argument('--aug_rotation', type=int, default=15,
                        help='max inplane rotation angle')
    parser.add_argument('--num_data_workers', type=int, default=12,
                        help='number of data loading workers, set according to the number of available CPU cores')
    parser.add_argument('--training_buffer_cpu', type=_strtobool, default=False, 
                        help='store training buffer on CPU memory instead of GPU, '
                        'this allows running ACE0 on smaller GPUs, but is slower')

    # === Registration parameters ==========================================================================================

    parser.add_argument('--ransac_iterations', type=int, default=32,
                        help="Number of RANSAC hypothesis when registering mapping frames.")
    parser.add_argument('--ransac_threshold', type=float, default=10,
                        help='RANSAC inlier threshold in pixels')

    # === Visualisation parameters =========================================================================================

    parser.add_argument('--render_visualization', type=_strtobool, default=False,
                        help="Render visualisation frames of the whole reconstruction process.")
    parser.add_argument('--render_flipped_portrait', type=_strtobool, default=False,
                        help="Dataset images are 90deg flipped (like Wayspots).")
    parser.add_argument('--render_marker_size', type=float, default=0.03,
                        help="Size of the camera marker when rendering scenes.")
    parser.add_argument('--iterations_output', type=int, default=500,
                        help='how often to print the loss and render a frame')

    parser.add_argument('--random_seed', type=int, default=1305,
                        help='random seed, predominately used to select seed images')

    opt = parser.parse_args()

    return opt
