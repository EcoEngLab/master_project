setwd("/home/jiayi-chen/Documents/MiCRM/code")
library(ggplot2)
library(readr)
library(tidyr)
library(dplyr)
library(stringr)
library(patchwork)

df <- read.csv("../data/df_results.csv")

long_data <- pivot_longer(
  df,
  cols = starts_with("RelAbun_Comm"), 
  names_to = c("Community", "Species"),
  names_pattern = "RelAbun_(Comm\\d+)_Sp(\\d+)", 
  values_to = "RelativeAbundance"
) %>%
  rowwise() %>%
  mutate(
    Community.CUE = get(paste0("Community.CUE.", str_remove(Community, "Comm"))),
    Species.CUE = get(paste0("CUE_Comm", str_remove(Community, "Comm"), "_Sp", Species))
  ) %>%
  ungroup() %>%
  select(-starts_with("Community.CUE."), -matches("^CUE_Comm"))
# standrdise species number
species_counts <- long_data %>%
  group_by(Community) %>%
  summarise(SpeciesCount = n_distinct(Species))

ggplot(long_data %>% filter(RelativeAbundance > 0), 
       aes(x = RelativeAbundance, fill = Community, color = Community)) +
  geom_density(alpha = 0.3, adjust = 2) +
  scale_x_log10() +
  theme_minimal() +
  labs(title = "Density of Relative Abundance by Community (log scale)",
       x = "Relative Abundance (log10 scale)", y = "Density") +
  scale_fill_manual(values = c("red", "green", "blue")) +
  scale_color_manual(values = c("red", "green", "blue"))

################### calculate the species richness ###################
long_data_d <- long_data %>%
  mutate(Status = ifelse(RelativeAbundance < 1e-5, "Extinction", "Survival"))

richness <- long_data_d %>%
  group_by(Seed, Community) %>%
  summarise(
   richness = sum(Status == "Survival"),
    TotalSpecies = n(),
    SurvivalRate =richness / TotalSpecies,
    .groups = "drop"
  )


shannon_evenness <- long_data_d %>%
  group_by(Seed, Community) %>%
  summarize(
  Shannon = -sum(RelativeAbundance[RelativeAbundance >0 ] * log(RelativeAbundance[RelativeAbundance >0 ])),
  Evenness = Shannon/log( sum(Status == "Survival")),
  )

# plot shannon and evenness
ggplot(shannon_evenness, aes(x = Community, y = Shannon, color = Community)) +
  geom_jitter(width = 0.2, height = 0, alpha = 0.7, size = 2) +
  theme_minimal() +
  labs(title = "Shannon Index per Community",
       x = "Community", y = "Shannon Index")
ggplot(shannon_evenness, aes(x = Community, y = Evenness, color = Community)) +
  geom_jitter(width = 0.2, height = 0, alpha = 0.7, size = 2) +
  theme_minimal() +
  labs(title = "Evenness per Community",
       x = "Community", y = "Evenness")

# plot relative richness
ggplot(richness, aes(x = Community, y = SurvivalRate, fill = Community)) +
  geom_boxplot(alpha = 0.6, outlier.shape = NA) +
  geom_jitter(width = 0.2, size = 1, alpha = 0.5) +
  theme_minimal() +
  labs(title = "Distribution of Relative Species Richness per Community",
       x = "Community", y = "Relative Richness (Survival Rate)")

