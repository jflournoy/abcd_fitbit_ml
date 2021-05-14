library(data.table)
library(lubridate)
data.table::setDTthreads(1)

aggregate_and_widen <- function(d){
  d[, fit_ss_date_posix := lubridate::mdy(fit_ss_date)]
  setorder(d, subjectkey, fit_ss_date_posix)
  d[, daynum := as.numeric(fit_ss_date_posix - fit_ss_date_posix[[1]]), by = 'subjectkey']
  d <- d[daynum <= 14]
  d[, c('fit_ss_date', 'fit_ss_date_posix', 'fit_ss_protocol_wear', 'fit_ss_wkno') := NULL]
  d_day_l <- melt(d, id.vars = c(idvars, 'daynum'))
  d_l <- copy(d_day_l)
  
  #Create day-level vars
  d_day_l[, c('_dxdt',
              '_devmean') := 
            list(value - shift(value, type = 'lag'),
                 value - mean(value, na.rm = TRUE)), 
          by = c('subjectkey', 'variable')]
  d_day_l_l <- melt(d_day_l, id.vars = c(idvars, 'daynum', 'variable'))
  d_day_l_l[, variable := paste0(variable, 
                                 fifelse(variable.1 == 'value', '', as.character(variable.1)), 
                                 sprintf('_d%02d', daynum))]
  
  d_day_l_l[, c('daynum', 'variable.1') := NULL]
  d_day_w <- dcast(d_day_l_l[!is.na(value)], ... ~ variable, value.var = 'value')
  
  #Create week-level vars
  d_l[, weekno := daynum %/% 7]
  d_each_week_l <- d_l[, list('mean' = mean(value, na.rm = TRUE),
                              'var' = sd(value, na.rm = TRUE)), by = c(idvars, 'weekno', 'variable')]
  d_each_week_l_l <- melt(d_each_week_l, id.vars = c(idvars, 'weekno', 'variable'))
  d_each_week_l_l[, variable := paste(variable, 
                                      variable.1, 
                                      sprintf('w%d', weekno), sep = '_')]
  d_each_week_l_l[, c('weekno', 'variable.1') := NULL]
  d_each_week_w <- dcast(d_each_week_l_l[!is.na(value)], ... ~ variable, value.var = 'value')
  
  #Create subject-level vars
  d_all_weeks_l <- d_l[, list('mean' = mean(value, na.rm = TRUE),
                              'var' = sd(value, na.rm = TRUE)), by = c(idvars, 'variable')]
  d_all_weeks_l_l <- melt(d_all_weeks_l, id.vars = c(idvars, 'variable'))
  d_all_weeks_l_l[, variable := paste(variable, 
                                      variable.1, sep = '_')]
  d_all_weeks_l_l[, c('variable.1') := NULL]
  d_all_weeks_w <- dcast(d_all_weeks_l_l[!is.na(value)], ... ~ variable, value.var = 'value')
  
  d_w <- merge(merge(d_day_w, d_each_week_w, all = TRUE), d_all_weeks_w, all = TRUE)
  return(d_w)
}


source('varlists.R')
idvars <- c('subjectkey', 'eventname', 'interview_age', 'sex')
# View(fread('~/data/ABCD/abcd_fbdss01_2d.txt')[eventname == '2_year_follow_up_y_arm_1' &
#                                                  fit_ss_protocol_wear == 1 & 
#                                                  subjectkey == 'NDAR_INVMWWHDNLT'])

fbdpas <- fread('~/data/ABCD/abcd_fbdpas01_2d.txt', 
                select = c(idvars, 
                           fbdpas_vars, 
                           'fit_ss_wear_date',
                           'fit_ss_wkno'))[eventname == '2_year_follow_up_y_arm_1' &
                                             fit_ss_protocol_wear == 1 & 
                                             fit_ss_wkno >= 0]
fbdss <- fread('~/data/ABCD/abcd_fbdss01_2d.txt', 
               select = c(idvars, 
                          fbdss_vars, 
                          'fit_ss_sleepdate',
                          'fit_ss_wkno'))[eventname == '2_year_follow_up_y_arm_1' &
                                            fit_ss_protocol_wear == 1 & 
                                            fit_ss_wkno >= 0]
cbcl <- fread('~/data/ABCD/abcd_cbcls01_2d.txt', 
              select = c(idvars, cbcl_vars))[eventname == '2_year_follow_up_y_arm_1']


sprintf('%d subjects without fitbit activity data', cbcl[!fbdpas, on = 'subjectkey'][, .N])
sprintf('%d subjects without fitbit sleep data', cbcl[!fbdss, on = 'subjectkey'][, .N])
sprintf('%d subjects with fitbit activity but not sleep data', unique(fbdpas[!fbdss, on = 'subjectkey'][, 'subjectkey'])[, .N])

setnames(fbdpas, 'fit_ss_wear_date', 'fit_ss_date')
setnames(fbdss, 'fit_ss_sleepdate', 'fit_ss_date')
fbdpas_w <- aggregate_and_widen(fbdpas)[, has_activity := 1]
fbdss_w <- aggregate_and_widen(fbdss)[, has_sleep := 1]
cbcl[, has_clinical := 1]
cbcl[, clinical := fifelse(cbcl_scr_syn_internal_t > 64 |
                             cbcl_scr_syn_external_t > 64 |
                             cbcl_scr_syn_totprob_t > 64, 1, 0)]
#cbcl[, sum(clinical == 1, na.rm = TRUE)/.N]

data_wide <- merge(cbcl, merge(fbdpas_w, fbdss_w, by = idvars, all = TRUE), by = idvars, all = TRUE)
head(names(data_wide))
system.time({
  print(
    hist(
      melt(data_wide)[, 
                      list(prop_missing = round(sum(is.na(value))/.N, 2)), 
                      by = 'variable'][, prop_missing],
      breaks = 100,
      xlab = 'Proportion missing values',
      main = 'Missing data by variable'))})
system.time({
  print(
    hist(
      melt(data_wide, 
           id.vars = 'subjectkey')[, 
                                   list(prop_missing = round(sum(is.na(value))/.N, 2)), 
                                   by = 'subjectkey'][, prop_missing], 
      breaks = 100,
      xlab = 'Proportion missing values',
      main = 'Missing data by subject'))})

fwrite(data_wide, file = 'cleaned_data.csv')
