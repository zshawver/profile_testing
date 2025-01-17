library(here)
library(openxlsx)
library(ggplot2)
library(wesanderson)
library(dplyr)

fn <- here("Data","FLOpioidsFG_Results_WeightTesting.xlsx")
sheet_names = getSheetNames(fn)

dat <- read.xlsx(fn,sheet = "results")

dat <- dat |> 
  filter(p_value <= .1) |> 
  mutate(logN = log(N))

p_dev = ggplot(data = dat, aes(x = normalized_dev, y = weight_zs))
p_dev + 
  geom_jitter(aes(color = type), alpha = 1.2)+
  scale_color_manual(values = wes_palette("Moonrise2")) +
  theme_minimal()

p_N = ggplot(data = dat, aes(x = N, y = weight_zs))
p_N + 
  geom_point(aes(color = type), alpha = 1.2)+
  scale_color_manual(values = wes_palette("Moonrise2")) +
  theme_minimal()

p_logN = ggplot(data = dat, aes(x = logN, y = weight_zs))
p_logN + 
  geom_point(aes(color = type), alpha = 1.2)+
  scale_color_manual(values = wes_palette("Moonrise2")) +
  theme_minimal()

p_weight_pred = ggplot(data = dat, aes(x = weight_zs, y = weighted_pred))
p_weight_pred+
  geom_point(aes(color = type), alpha = 1.2)+
  scale_color_manual(values = wes_palette("Moonrise2")) +
  theme_minimal()
