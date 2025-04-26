library(data.table)
library(tidyverse)
library(ggplot2)
library(here)

dat <- fread(here("simulations", "test_f1_results.csv"))

dat[, seed := as.numeric(str_extract(filename, "(?<=_results_)\\d+"))]
dat[, model := str_extract(filename, "(?<=ag_news_)[^_]+")]

p <- ggplot(dat, aes(x = id, y = test_f1, color = filename)) +
  geom_point() +
  geom_line() +
  theme_minimal() +
  labs(x = "ID", y = "Test F1")

p

dat2 <- dat[, .(test_f1 = mean(test_f1)), by = .(id, model)]

p2 <- ggplot(dat2, aes(x = id, y = test_f1, color = model)) +
  geom_point() +
  geom_line() +
  theme_minimal() +
  labs(x = "ID", y = "Test F1")

p2

