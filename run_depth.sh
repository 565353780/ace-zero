cd ../ace0

# --prior_loss_type laplace_nll \

python ace_zero.py \
  '/home/lichanghao/chLi/Dataset/GS/haizei_1/images/*.png' \
  '/home/lichanghao/chLi/Dataset/GS/haizei_1/ace0_depth/' \
  --loss_structure probabilistic \
  --prior_loss_type laplace_wd \
  --prior_loss_weight 0.1 \
  --prior_loss_bandwidth 0.6 \
  --prior_loss_location 1.73
