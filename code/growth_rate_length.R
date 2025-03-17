setwd("/home/jiayi-chen/Documents/MiCRM/code")
library(ggplot2)
# Read the CSV file into R
df <- read.csv("../data/growth rate length vs. CUE.csv")
model1<- lm(Community.CUE.1 ~ Length.r1, data = df)
model2<- lm(Community.CUE.2 ~ Length.r2, data = df)
model3<- lm(Community.CUE.1 ~ Length.r1, data = df)
summary(model1)
summary(model2)
summary(model3)
p <- ggplot(df, aes(x = Length.r1, y = Community.CUE.1))+
  geom_point(color = "blue", alpha = 0.6, size = 2) +
  geom_smooth(method = "lm", color = "blue", se = FALSE) +  
  geom_point(aes(x = Length.r2, y = Community.CUE.2), color = "red", alpha = 0.6, size = 2) +
  geom_smooth(aes(x = Length.r2, y = Community.CUE.2), method = "lm", color = "red", se = FALSE) +
  geom_point(aes(x = Length.r3, y = Community.CUE.3), color = "green", alpha = 0.6, size = 2) +
  geom_smooth(aes(x = Length.r3, y = Community.CUE.3), method = "lm", color = "green", se = FALSE) +
  labs(title = "Length of growth rate vs. CUE",
       x = "Length_r",
       y = "Community CUE") +
  theme_minimal()
print(p)
