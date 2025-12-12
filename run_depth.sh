cd ../ace0

# --prior_loss_type laplace_nll \

python ace_zero.py \
  '/home/chli/chLi/Dataset/GS/test1/images/*.png' \
  '/home/chli/chLi/Dataset/GS/test1/ace0/' \
  --loss_structure probabilistic \
  --prior_loss_type laplace_wd \
  --prior_loss_weight 0.1 \
  --prior_loss_bandwidth 0.6 \
  --prior_loss_location 1.73 \
  --render_visualization True
