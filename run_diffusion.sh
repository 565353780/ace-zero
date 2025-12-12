cd ../ace0

python ace_zero.py \
  '/home/chli/chLi/Dataset/GS/test1/images/*.png' \
  '/home/chli/chLi/Dataset/GS/test1/ace0/' \
  --loss_structure "dsac*" \
  --prior_loss_type diffusion \
  --prior_loss_weight 200 \
  --prior_diffusion_model_path /home/lichanghao/chLi/Model/ACE0/diffusion_prior.pt \
  --render_visualization True
