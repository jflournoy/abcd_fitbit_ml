library(data.table)
d_ <- data.table(expand.grid(mode = c('pa', 'sleep'), s = c('daily', 'weekly', 'id'), outcome = c('subscale', 'scale', 'overall')))
d_split <- split(d_, d_[, c('mode', 's', 'outcome')])
#x <- d_split[[1]]
d <- data.table::rbindlist(lapply(d_split, function(x){
  if (x[, s] == 'daily'){
    x <- cbind(x, data.table(time = c(7, 14, 21)))
  } else if (x[, s] == 'weekly') {
    x <- cbind(x, data.table(time = c(1, 2, 3)))
  } else {
    x
  }
}), fill = TRUE)
d[is.na(time), time := 0]
d

readr::write_tsv(d, file = '~/code/abcd_ml/arguments_file.txt', col_names = FALSE)
