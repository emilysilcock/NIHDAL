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


dat[, mean(target_religion)]
dat[, mean(target_religion_atheist)]
dat[, mean(target_religion_buddhist)]
dat[, mean(target_religion_christian)]
dat[, mean(target_religion_hindu)]
dat[, mean(target_religion_jewish)]
dat[, mean(target_religion_mormon)]
dat[, mean(target_religion_muslim)]
dat[, mean(target_religion_other)]


dat[target_religion == 1, mean(target_religion_atheist)]
dat[target_religion == 1, mean(target_religion_buddhist)]
dat[target_religion == 1, mean(target_religion_christian)]
dat[target_religion == 1, mean(target_religion_hindu)]
dat[target_religion == 1, mean(target_religion_jewish)]
dat[target_religion == 1, mean(target_religion_mormon)]
dat[target_religion == 1, mean(target_religion_muslim)]
dat[target_religion == 1, mean(target_religion_other)]

dat[target_religion == 1, mean(target_religion_atheist + target_religion_buddhist + target_religion_christian + target_religion_hindu + target_religion_jewish + target_religion_mormon + target_religion_muslim + target_religion_other)]

View(dat[target_religion == 1, .(comment_id, annotator_id, text, hatespeech, hate_speech_score)])
View(dat[target_religion_hindu == 1 & hate_speech_score > 1, .(comment_id, annotator_id, text, hatespeech, hate_speech_score)])

foo <- dat[, .(n = .N,
              hatespeech = mean(hatespeech),
              hate_speech_score = mean(hate_speech_score),
              min_hate_speech_score = min(hate_speech_score),
              max_hate_speech_score = max(hate_speech_score),
              target_religion = mean(target_religion),
              target_religion_atheist = mean(target_religion_atheist),
              target_religion_buddhist = mean(target_religion_buddhist),
              target_religion_christian = mean(target_religion_christian),
              target_religion_hindu = mean(target_religion_hindu),
              target_religion_jewish = mean(target_religion_jewish),
              target_religion_mormon = mean(target_religion_mormon),
              target_religion_muslim = mean(target_religion_muslim),
              target_religion_other = mean(target_religion_other)),
          by = .(comment_id, text)]

foo[, label := ifelse(hate_speech_score > 1 & target_religion == 1, 1, 0)]

foo[, sum(label == 1)]
foo[, sum(label == 1 & target_religion_atheist == 0 & target_religion_buddhist == 0 & target_religion_christian == 0 & target_religion_hindu == 0 & target_religion_jewish == 0 & target_religion_mormon == 0 & target_religion_muslim == 0 & target_religion_other == 0)]

fwrite(foo, here("data", "hate_speech_dataset_summary.csv"))

summary(foo$n)
View(foo[n == 815])

View(foo[hatespeech > 1 & target_religion_hindu > 0, .(text, hatespeech, hate_speech_score, target_religion_hindu)])

foo[, sum(hate_speech_score > 1 & target_religion > 0)]
foo[, sum(hate_speech_score > 1 & hatespeech > 1 & target_religion > 0)]
foo[, sum(hate_speech_score > 1 & target_religion == 1)]
foo[, sum(hate_speech_score > 1 & target_religion == 1 & hatespeech > 1)]

foo[, sum(hate_speech_score > 1 & target_religion_christian > 0)]
foo[, sum(hate_speech_score > 1 & target_religion_christian == 1)]

foo[, sum(hate_speech_score > 1 & target_religion_atheist > 0)]
foo[, sum(hate_speech_score > 1 & target_religion_atheist == 1)]

foo[, sum(hate_speech_score > 1 & target_religion_buddhist > 0)]
foo[, sum(hate_speech_score > 1 & target_religion_buddhist == 1)]

foo[, sum(hate_speech_score > 1 & target_religion_jewish > 0)]
foo[, sum(hate_speech_score > 1 & target_religion_jewish == 1)]

foo[, sum(hate_speech_score > 1 & target_religion_mormon > 0)]
foo[, sum(hate_speech_score > 1 & target_religion_mormon == 1)] 

foo[, sum(hate_speech_score > 1 & target_religion_hindu > 0)]
foo[, sum(hate_speech_score > 1 & target_religion_hindu == 1)]

foo[, sum(hate_speech_score > 1 & target_religion_muslim > 0)]
foo[, sum(hate_speech_score > 1 & target_religion_muslim == 1)] 

foo[, sum(hate_speech_score > 1 & target_religion_other > 0)]
foo[, sum(hate_speech_score > 1 & target_religion_other == 1)]




View(foo[hate_speech_score > 1 & target_religion_christian > 0, .(text, hatespeech, hate_speech_score)])




View(dat[target_religion_jewish == 1 & hatespeech > 1, .(comment_id, annotator_id, text, hatespeech)])
View(dat[target_religion_jewish == 1 & hate_speech_score > 1, .(comment_id, annotator_id, text, hatespeech)])

