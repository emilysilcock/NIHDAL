library(data.table)
library(tidyverse)
library(here)



dat <- fread(here("data", "hate_speech_dataset.csv"))


setorder(dat, comment_id, annotator_id)

View(dat[, .(comment_id, annotator_id, text, hatespeech)])

dat[, mean(target_race)]
dat[, mean(target_race_asian)]
dat[, mean(target_race_black)]
dat[, mean(target_race_latinx)]
dat[, mean(target_race_middle_eastern)]
dat[, mean(target_race_native_american)]
dat[, mean(target_race_pacific_islander)]
dat[, mean(target_race_white)]
dat[, mean(target_race_other)]

dat[, sum(target_race)]
dat[, sum(target_race_asian)]
dat[, sum(target_race_black)]
dat[, sum(target_race_latinx)]
dat[, sum(target_race_middle_eastern)]
dat[, sum(target_race_native_american)]
dat[, sum(target_race_pacific_islander)]
dat[, sum(target_race_white)]
dat[, sum(target_race_other)]


View(dat[target_race_pacific_islander == 1, .(comment_id, annotator_id, text, hatespeech)])


dat[, .N, hatespeech]

hist(dat$hate_speech_score)

dat[, mean(hatespeech == 2)]
dat[, mean(hate_speech_score > 2)]


View(dat[hate_speech_score > 2, .(comment_id, annotator_id, text, hatespeech)])

th <- 1

dat[, mean(target_race & hate_speech_score > th)]
dat[, mean(target_race_asian & hate_speech_score > th)]
dat[, mean(target_race_black & hate_speech_score > th)]
dat[, mean(target_race_latinx & hate_speech_score > th)]
dat[, mean(target_race_middle_eastern & hate_speech_score > th)]
dat[, mean(target_race_native_american & hate_speech_score > th)]
dat[, mean(target_race_pacific_islander & hate_speech_score > th)]
dat[, mean(target_race_white & hate_speech_score > th)]
dat[, mean(target_race_other & hate_speech_score > th)]
