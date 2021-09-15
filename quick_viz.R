library(data.table)
library(ggplot2)
library(patchwork)
library(viridis)
setDTthreads(4)
d_ <- fread('train_data.csv')
d <- unique(d_[, c('subjectkey', 'interview_age', 'sex', 'cbcl_scr_syn_totprob_t')])
summary(lm(cbcl_scr_syn_totprob_t ~ interview_age*sex, data = d))

ggplot(d, aes(x = interview_age, y = cbcl_scr_syn_totprob_t)) + 
  geom_hex(aes(color = ..count.., fill = ..count..))

ggplot(d, aes(x = sex, y = cbcl_scr_syn_totprob_t)) + 
  geom_violin() + 
  geom_boxplot(width = .25)


cbcl <- fread('~/data/ABCD/abcd_cbcls01_2d.txt')[eventname == '2_year_follow_up_y_arm_1']

summary(lm(cbcl_scr_syn_totprob_t ~ interview_age*sex, data = cbcl))

ggplot(cbcl, aes(x = interview_age, y = cbcl_scr_syn_totprob_t)) + 
  geom_hex(aes(color = ..count.., fill = ..count..), binwidth = c(4,4)) + 
  geom_line(stat = 'smooth', method = 'gam', formula = y ~ s(x), color = 'red') + 
  scale_color_viridis(lim = c(0, 150), aesthetics = c('color', 'fill')) + 
  labs(title = 'CBCL') + 
  theme_minimal() + 
ggplot(d, aes(x = interview_age, y = cbcl_scr_syn_totprob_t)) + 
  geom_hex(aes(color = ..count.., fill = ..count..), binwidth = c(4,4)) + 
  geom_line(stat = 'smooth', method = 'gam', formula = y ~ s(x), color = 'red') + 
  scale_color_viridis(lim = c(0, 150), aesthetics = c('color', 'fill')) + 
  labs(title = 'Training') + 
  theme_minimal() + 
  plot_layout(guides = 'collect')

ggplot(cbcl, aes(x = sex, y = cbcl_scr_syn_totprob_t)) + 
  geom_violin() + 
  geom_boxplot(width = .25) + 
  labs(title = 'CBCL') + 
  theme_minimal() +
ggplot(d, aes(x = sex, y = cbcl_scr_syn_totprob_t)) + 
  geom_violin() + 
  geom_boxplot(width = .25) + 
  labs(title = 'Training') + 
  theme_minimal()

set.seed(9003665)
d_ben <- d_[unique(d_[,'subjectkey'])[, .(subjectkey = sample(subjectkey, 200))], on = 'subjectkey']
fwrite(d_ben, file = 'ben.csv')

cbcl[unique(d_ben[, c('subjectkey', 'sex', 'interview_age')]), on = 'subjectkey'][, c('subjectkey', 'sex', 'interview_age', 'i.sex', 'i.interview_age')]

