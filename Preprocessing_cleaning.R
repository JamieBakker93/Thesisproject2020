## Loading the different libraries----------------------------------------------
install.packages("sqldf")
install.packages("magrittr")
install.packages("dplyr")
library(ggplot2)
library(tidyverse)
library(sqldf)
library(stats)
library(dplyr)
library(lattice)
library(caret)
library(lubridate)
library(reshape2)
library(tidyr)
library(purrr)
library(magrittr)
require(data.table)
## setting work directions------------------------------------------------------ 
setwd("C:/Users/jamie/Dropbox/School/Master Data Science/Thesis/master_student_phone_use_data")

## loading the 3 different datasets---------------------------------------------
df1 <- read.csv("app_categories.csv", stringsAsFactors =  FALSE, sep = ",",  na.strings=c("","NA")) 
df2 <- read.csv("phone_use_data.csv", stringsAsFactors =  FALSE, sep = ",",  na.strings=c("","NA"))  
df3 <- read.csv("mood_sampling_data.csv", stringsAsFactors =  FALSE, sep = ",", na.strings=c("","NA"))

## Remove duplicate Rows--------------------------------------------------------
df3 <- distinct(df3)
df2 <- distinct (df2)

############ MOOD DATASET CLEANING AND PREPPING ################################

## Checking and removing user_ids that are not a pair in the different datasets-
list_phone <- c(as.character(unique(df2$user_id)))
list_mood  <- c(as.character(unique(df3$user_id)))
list_mood[!(list_mood %in% list_phone)]
list_phone[!(list_phone %in% list_mood)]

df3 <- df3[df3[["user_id"]] != "3220",]
df3 <- df3[df3[["user_id"]] != "5343",]
df3 <- df3[df3[["user_id"]] != "5350",]
df3 <- df3[df3[["user_id"]] != "5363",]
df3 <- df3[df3[["user_id"]] != "5378",]
df3 <- df3[df3[["user_id"]] != "5387",]
df3 <- df3[df3[["user_id"]] != "5421",]
df3 <- df3[df3[["user_id"]] != "5440",]
df3 <- df3[df3[["user_id"]] != "5441",]
df3 <- df3[df3[["user_id"]] != "7372",]
df3 <- df3[df3[["user_id"]] != "7373",]
df3 <- df3[df3[["user_id"]] != "10175",]
df3 <- df3[df3[["user_id"]] != "10194",]
df3 <- df3[df3[["user_id"]] != "10199",]
df3 <- df3[df3[["user_id"]] != "10217",]
df3 <- df3[df3[["user_id"]] != "10218",]
df3 <- df3[df3[["user_id"]] != "10249",]
df3 <- df3[df3[["user_id"]] != "10252",]
df3 <- df3[df3[["user_id"]] != "10271",]
df3 <- df3[df3[["user_id"]] != "10320",]
df3 <- df3[df3[["user_id"]] != "10592",]
df3 <- df3[df3[["user_id"]] != "10602",]
df3 <- df3[df3[["user_id"]] != "10603",]
df3 <- df3[df3[["user_id"]] != "10607",]
df3 <- df3[df3[["user_id"]] != "10913",]

## Removing all untrusthy entries Blocked, canceled or expired, Unknown---------
unique(df3$duration)

## Remove Expired elements
df3 <- df3[df3[["duration"]] != "Expired",]
## Remove Blocked elements
df3 <- df3[df3[["duration"]] != "Blocked",]
## Remove Canceled elements
df3 <- df3[df3[["duration"]] != "Canceled",]
## Removing Unknown elements
df3 <- df3[df3[["duration"]] != "Unknown",]

## Remove a totally empty ROW with ROW number-----------------------------------
df3 <- df3[-c(4653),]

## Changing the mood variables to another scale---------------------------------

old <- 0:5
new <- 1:6

df3$anxious <- as.integer(as.character(factor(df3$anxious, old, new)))
df3$bored <- as.integer(as.character(factor(df3$bored, old, new)))
df3$gloomy <- as.integer(as.character(factor(df3$gloomy, old, new)))
df3$stressed <- as.integer(as.character(factor(df3$stressed, old, new)))
df3$calm <- as.integer(as.character(factor(df3$calm, old, new)))
df3$content <- as.integer(as.character(factor(df3$content, old, new)))
df3$cheerful <- as.integer(as.character(factor(df3$cheerful, old, new)))
df3$tired <- as.integer(as.character(factor(df3$tired, old, new)))
df3$energetic <- as.integer(as.character(factor(df3$energetic, old, new)))
df3$upset <- as.integer(as.character(factor(df3$upset, old, new)))
df3$envious <- as.integer(as.character(factor(df3$envious, old, new)))
df3$inferior <- as.integer(as.character(factor(df3$inferior, old, new)))
df3$energetic <- as.integer(as.character(factor(df3$energetic, old, new)))
df3$enjoy <- as.integer(as.character(factor(df3$enjoy, old, new)))

## Removing all variables that are not needed for this thesis-------------------
#df3 <- df3[-c(19:34)]

## Making new variables to have an outcome variable-----------------------------

## making 6 new variable to stress / energy and selfesteem----------------------


## new energy variables
df3$energyNega <- rowMeans(df3[,c(6,7,12)], na.rm=TRUE)
df3$energyPosi <- rowMeans(df3[,c(11,13)], na.rm=TRUE)

old <- 1:6
new <- 6:1

df3$energyNega <- cut(df3$energyNega, 
                                  breaks=c(-Inf,1.5, 2.5, 3.5, 4.5 ,5.5, Inf), 
                                  labels=c(1,2,3,4,5,6 ))

df3$energyNega <- as.integer(as.character(factor(df3$energyNega, old, new)))


## Make 3 new variables positive variable - negative variable 
## 1 means low stress, high energy level, high self esteem 


df3$energylevel     <- rowMeans(df3[,c(36,37)], na.rm=TRUE)

#df3$energylevel <- cut(df3$energylevel, 
                      #breaks=c(-Inf,1.5, 2.5, 3.5, 4.5 ,5.5, Inf), 
                      #labels=c(1,2,3,4,5,6 ))


df3$energylevel <- cut(df3$energylevel, 
                       breaks=c(-Inf, 3.5, 4.5, Inf), 
                       labels=c(1,2,3))

## Remove variables that are not needed anymore
df3 <- df3[-c(2,4,17:18, 20:34,36,37)]

## Quick check if we have NA values after making the changes -------------------
new_DF <- df3[rowSums(is.na(df3)) > 0,]
remove(new_DF)

## making new date and time variables
df3$date <- (substring(df3$response_time ,0,10))
df3$time <- (substring(df3$response_time,12,20))
df3$hours <- as.numeric(substring(df3$time,0,2))
df3$minutes <- as.numeric(substring(df3$time,4,5))
df3$seconds <- as.numeric(substring(df3$time,7,8))
df3$timeinseconds <- (df3$hours*3600 + df3$minutes*60 + df3$seconds)
df3 <- df3[,c(1:19,23)]
df3$user_date <- paste(df3$date, df3$user_id)

## seperate the four different moods based on time window ---------------------
df3.1 <- df3[df3[["day_time_window"]] == "1",]
df3.2 <- df3[df3[["day_time_window"]] == "2",]
df3.3 <- df3[df3[["day_time_window"]] == "3",]
df3.4 <- df3[df3[["day_time_window"]] == "4",]


## Combine the datasets so you have have the moods together to predict----------
combine1.2 <- merge(df3.1, df3.2, by = "user_date")
combine2.3 <- merge(df3.2, df3.3, by = "user_date")
combine3.4 <- merge(df3.3, df3.4, by = "user_date")
combine_all <- rbind(combine1.2, combine2.3, combine3.4)

## Remove all the mood variables for later moodset------------------------------
#combine_all <- combine_all[,c(1:23,25,28,44,45)]

## See if there is at least two hours between the two mood surveys--------------
combine_all$timedifference <- combine_all$timeinseconds.y -
  combine_all$timeinseconds.x

## Remove all rows with less than 2 hours in between 
MOOD <-  combine_all#[combine_all["timedifference"] > 7200,]

remove(df3.1, df3.2, df3.3, df3.4, combine1.2, combine2.3, combine3.4, combine_all)

MOOD <- MOOD[,c(1:21, 38:42)]

names(MOOD)[names(MOOD) == "user_id.x"] <- "user_id"
names(MOOD)[names(MOOD) == "response_time.x"] <- "response_time_1"
names(MOOD)[names(MOOD) == "time.x"] <-  "time_1" 
names(MOOD)[names(MOOD) == "date.x"] <-  "date_1" 
names(MOOD)[names(MOOD) == "anxious.x"] <-  "Anxious"
names(MOOD)[names(MOOD) == "timeinseconds.x"] <- "Time_in_seconds_1"
names(MOOD)[names(MOOD) == "bored.x"] <-  "Bored"
names(MOOD)[names(MOOD) == "gloomy.x"] <-   "Gloomy"
names(MOOD)[names(MOOD) == "calm.x"] <-  "Calm"
names(MOOD)[names(MOOD) == "stressed.x"] <-  "Stressed"
names(MOOD)[names(MOOD) == "content.x"] <-  "Content"
names(MOOD)[names(MOOD) == "cheerful.x"] <-   "Cheerful"
names(MOOD)[names(MOOD) == "tired.x"] <-  "Tired"
names(MOOD)[names(MOOD) == "energetic.x"] <- "Energetic"
names(MOOD)[names(MOOD) == "upset.x"] <-  "Upset"
names(MOOD)[names(MOOD) == "envious.x"] <-  "Envious"
names(MOOD)[names(MOOD) == "inferior.x"] <-  "inferior"
names(MOOD)[names(MOOD) == "activity.x"] <-  "activity"
names(MOOD)[names(MOOD) == "social.x"] <-  "social"
names(MOOD)[names(MOOD) == "enjoy.x"] <-  "enjoy"
names(MOOD)[names(MOOD) == "energylevel.x"] <-  "Energy_level"
names(MOOD)[names(MOOD) == "energylevel.y"] <-  "Outcome_variable"
names(MOOD)[names(MOOD) == "response_time.y"] <- "response_time_2"
names(MOOD)[names(MOOD) == "date.y"] <-   "date_2"
names(MOOD)[names(MOOD) == "time.y"] <-  "time_2"
names(MOOD)[names(MOOD) == "timeinseconds.y"] <-  "Time_in_seconds_2"

MOOD <- MOOD[,c(1,2,3,19,20,21,4:18,23:26,22)]

   

########### PHONE DATASET CLEANING AND PREPPING ################################

##Combining the phone and category dataframes----------------------------------
names(df1)[names(df1) == "app_id"] <- "application"

df4<- sqldf("select * from df2 
              left join df1 
              on df2.application = df1.application", row.names = TRUE)


##analysing the apps without category (NAs)
NoCategory <- df4$application[!(df4$application %in% df1$application)]
NoCategory <- as.data.frame(table(NoCategory))
NoCategory <- (NoCategory %>% arrange(desc(Freq)))[1:2000,]

## adding the top 20 without NA in categories ---------------------------------
names(NoCategory)[names(NoCategory) == "NoCategory"] <- "application"
names(NoCategory)[names(NoCategory) == "Freq"] <- "count"

NoCategory$name <- c("Android System", "Ethica Logger", "Paralel Space", "Osiris Tilburg",
          "Wild West Craft","Napster", "Nexus Launcher","Deliveroo", "MO PTT", 
          "Asterix and Friends", "Takeaway", "Math Calculator","Keyboard Theme", "Quizkampen",
          "GMX Email", "Kakoa Talk", "Keyboard Theme","Keyboard Theme", "Voetbal Inside","Keyboard Theme" )

NoCategory$category <- c("Background Process", "Tools", "Tools", "Education", 
                         "Action", "Music & Audio", "Personalization", "Food & Drink",
                         "Social","Action","Food & Drink", "Tools", "Personalization",
                         "Trivia","Productivity","Social", "Personalization", 
                         "Personalization","News & Magazines", "Personalization")
  
NoCategory$better_category_hybrid <- c("Background Process", "Ethica", "None", "Education",
                                      "Game_Multiplayer", "None", "None", "None", "None",
                                      "Game_Multiplayer","None","Phone_Tools","None",
                                      "Game_Singleplayer","Email","None", "None", "None", "news"
                                      ,"None")
  
NoCategory$better_category <- c("Background Process", "Ethica", "Phone_Tools", "Education",
                                "Game_Singleplayer","Streaming_Services","Phone_Personalization",
                               " Online_Shopping","Social_Networking" ,"Game_Multiplayer"
                               ,"Online_Shopping", "Phone_Tools","Phone_Personalization",
                               "Game_Singleplayer","Email" ,"Social_Networking","Phone_Personalization"
                               , "Phone_Personalization","News","Phone_Personalization" )

df1 <- rbind(df1, NoCategory)  

## adding the top 10 apllications as seperate Categories -----------------------
NewCategory <- as.data.frame(table(df4$application))
names(NewCategory)[names(NewCategory) == "Var1"] <- "application"
names(NewCategory)[names(NewCategory) == "Freq"] <- "count"
NewCategory <- NewCategory[NewCategory[["application"]] != "com.ethica.logger",]
NewCategory <- NewCategory[NewCategory[["application"]] != "com.android.systemui",]
NewCategory <- data.table(NewCategory, key="count")
NewCategory <- NewCategory[, tail(.SD, 10),]


NewCategory$name <- c( "Settings", "Google Search", "Facebook Messenger", "YouTube", "Spotify",
                      "Facebook", "Google Chrome","Snapchat", "Instagram" ,  "WhatsApp Messenger")
NewCategory$category <- c("Settings", "Google Search", "Facebook Messenger", "YouTube", "Spotify",
                           "Facebook", "Google Chrome","Snapchat", "Instagram" , "WhatsApp Messenger")
NewCategory$better_category_hybrid <- c("None", "None","None", "None","None",
                                        "None","None", "None","None", "None")
NewCategory$better_category <-  c( "Settings" , "Google Search", "Facebook Messenger", "YouTube", "Spotify",
                                    "Facebook", "Google Chrome","Snapchat", "Instagram" , "WhatsApp Messenger")
df1 <- df1[-c(1:6, 10, 14, 15,19), ]
df1 <- rbind(df1, NewCategory,fill=TRUE)  

## make a new mood with the 20 apps in -----------------------------------------
df4<- sqldf("select * from df2 
            left join df1 
            on df2.application = df1.application", row.names = TRUE)

## Remove the background process------------------------------------------------
df4 <- df4[df4[["application"]] != "com.ethica.logger",]
df4 <- df4[df4[["category"]] != "Background Process",]

# Changing categories to better categories 
categories <- as.data.frame(table(df4$better_category))
df4$better_category[df4$better_category == "Messages"] <- "Messaging"
df4$better_category[df4$better_category == "Portfolio/Trading"] <- "Business_Management"
df4$better_category[df4$better_category == "Job_Search"] <- "Social_Networking"

df4$startTime <- ymd_hms(df4$startTime)
df4$endTime <- ymd_hms(df4$endTime)
df4$duration <- as.numeric(difftime(df4$endTime,df4$startTime ,units = "secs"))

## Creating a new data file with filtered top 10 applications used--------------
df4.1 <- df4
df4.1 <- filter(df4.1, df4.1$name == "Google Search"|
                  df4.1$name == "Messages"  |
                  df4.1$name == "Phone"|
                  df4.1$name == "Gmail" |
                  df4.1$name == "Snapchat" |
                  df4.1$name == "Instagram" |
                  df4.1$name == "Facebook Messenger" |
                  df4.1$name == "Google Chrome" |
                  df4.1$name == "Facebook"|
                  df4.1$name == "WhatsApp Messenger" )



table(df4$better_category)
## Combining categories --------------------------------------------------------
df4$better_category[df4$better_category == "Personal_Fitness"] <- "Sports"
df4$better_category[df4$better_category == "Personal_Fitness"] <- "Sports"

df4$better_category[df4$better_category == "Phone_Optimization"] <- "Phone_Tools"
df4$better_category[df4$better_category == "Phone_Personalization"] <- "Phone_Tools"
df4$better_category[df4$better_category == "Security"] <- "Phone_Tools"
df4$better_category[df4$better_category == "Maps"] <- "Phone_Tools"
df4$better_category[df4$better_category == "Settings"] <- "Phone_Tools"
df4$better_category[df4$better_category == "Time_Tracker"] <- "Phone_Tools"
df4$better_category[df4$better_category == "To-Do_List"] <- "Phone_Tools"
df4$better_category[df4$better_category == "Calendar"] <- "Phone_Tools"
df4$better_category[df4$better_category == "Weather"] <- "Phone_Tools"
df4$better_category[df4$better_category == "Phone_Assistant"] <- "Phone_Tools"

df4$better_category[df4$better_category == "Coupons"] <- "Shopping"
df4$better_category[df4$better_category == " Online_Shopping"] <- "Shopping"
df4$better_category[df4$better_category == "Online_Shopping "] <- "Shopping"

df4$better_category[df4$better_category == "Dialer"] <- "Phone"
df4$better_category[df4$better_category == "Dating"] <- "Social_Networking"

df4$better_category[df4$better_category == "Video_Players_&_Editors"] <- "Entertainment"
df4$better_category[df4$better_category == "Streaming_Services"] <- "Entertainment"
df4$better_category[df4$better_category == "Music_&_Audio"] <- "Entertainment"
df4$better_category[df4$better_category == "Book_Readers"] <- "Entertainment"
df4$better_category[df4$better_category == "Drawing"] <- "Entertainment"
df4$better_category[df4$better_category == "Auto_&_Vehicles"] <- "Entertainment"
df4$better_category[df4$better_category == "Game_Multiplayer"] <- "Entertainment"
df4$better_category[df4$better_category == "Game_Singleplayer"] <- "Entertainment"

df4$better_category[df4$better_category == "Messaging"] <- "Instant_Messaging"

df4$better_category[df4$better_category == "Education"] <- "Office"
df4$better_category[df4$better_category == "Business_Management"] <- "Office"
df4$better_category[df4$better_category == "Document_Editor"] <- "Office"

df4$better_category[df4$better_category == "Home_Automation"] <- "Other"
df4$better_category[df4$better_category == "Personal_Finance"] <- "Other"
df4$better_category[df4$better_category == "Medical"] <- "Other"
df4$better_category[df4$better_category == "Food_&_Drinks"] <- "Other"
df4$better_category[df4$better_category == "Wearables"] <- "Other"




table(df4$better_category)
## Getting top 25 categories used ---------------------------------------------
categories25 <- as.data.frame(table(df4$better_category))
names(categories25)[names(categories25) == "Var1"] <- "Categories"
names(categories25)[names(categories25) == "Freq"] <- "Frequenties"
categories25 <- data.table(categories25, key="Frequenties")
categories25 <- categories25[, tail(.SD, 35),]

##Insert Graph


##Filter only top 25 categories -----------------------------------------------


df4 <- df4[c(15,9,7,8,3,4,16)]

df5 <- as.data.table(df4)
df5 <- df5[-c(4,6)]

df5$date <- (substring(df5$startTime,0,10))
df5$time <- (substring(df5$startTime,12,20))
df5$hours <- as.numeric(substring(df5$time,0,2))
df5$minutes <- as.numeric(substring(df5$time,4,5))
df5$seconds <- as.numeric(substring(df5$time,7,8))
df5$timeinseconds <- (df5$hours*3600 + df5$minutes*60 + df5$seconds)
#df3 <- df3[,c(1,2,19,20,24, 3:18)]
df5$user_date <- paste(df5$date, df5$user_id)

## JE MOET DIT STUK EERST RUNNEN EN DAN WEER HASHTAGS ER VOOR ZETTEN WANT JE MOET EEN BEGIN DATAFRAME HEBBEN
df6 <- df5[df5[["user_date"]] == "2019-03-07 10597",]
df6 <- df6[df6[["timeinseconds"]] > "37026",]
df6 <- df6[df6[["timeinseconds"]] < "45203",]
d66 <- df6
df6 <- df6[ , .(duration = (duration)), by = .(better_category, user_date)]

df7 <- df6 %>% 
  group_by( user_date,better_category) %>% 
  summarise_all(sum)

df8 <- df7[-c(1)]
df8 <- t(df8)
colnames(df8) <- as.character(unlist(df8[1,]))
df8 = df8[-1, ]
df8 <- as.data.table(t(df8))
df8$user_date <- "2019-03-07 10597"

a =df8
a$timeframe <- MOOD[1087,22]



### STAP 1 ####################################################################
stap1 <- function(X, Y){
  X = (bind_rows(Y,X))
  return(X)}

stap2 <- function (X,Y) {
  X <- X[-c(2)]
  X <- t(X)
  colnames(X) <- as.character(unlist(X[1,]))
  X = X[-1, ]
  X <- as.data.table(t(X))
  X$user_date <- Y[1,1]
  return(X)}
  
stap3 <- function(X){
  X <- X %>% 
    group_by(better_category, user_date) %>% 
    summarise_all(sum)
  return(X)}

stap4 <- function(X){
  Y <- data.table(X)
  Y <- Y[ , .(Totalcount = (duration)), by = .(better_category, user_date)]
  return(Y)}

stap5 <- function(X,Y,Z){
  X <- X[X[["user_date"]]== Y[1,1],]
  X <- X[X[["timeinseconds"]] > Y[1,6],]
  X <- X[X[["timeinseconds"]] < Y[1,24],]
  if (length(X$user_date) == 0) {
    return(Z)} 
  else {return(X)}}

stap6 <- function(X,Y,Z, O){
  A <- stap5(X,Y,O)
  B <- stap4(A)
  C <- stap3(B)
  D <- stap2(C,Y)
  E <- stap1(Z,D)
  return(E)}


MOOD1 <- MOOD
FinalPhone <- a

repeat { 
  FinalPhone  <- stap6(df5, MOOD1, FinalPhone ,d66)
  #probeersel$timeframe <- MOOD1[1,19]
  MOOD1 <-MOOD1[-c(1),]
  if(nrow(MOOD1) == 0) {
    break}}


FinalPhone <- FinalPhone[-c(1)]
FinalPhone$timeframe <- MOOD[,20]
FinalPhone$user_date_time <- paste(FinalPhone$user_date,FinalPhone$timeframe)
MOOD$user_date_time <- paste(MOOD$user_date, MOOD$day_time_window.x)
MOODPHONE <- merge(FinalPhone, MOOD, by = "user_date_time")
MOODPHONE[is.na(MOODPHONE)] <- 0
MOODPHONE <- distinct(MOODPHONE)
MOODPHONE <- MOODPHONE[,c(1:8,10, 12:28,36:48,50, 55)]
MOODPHONE[,2:26] <- as.data.frame(sapply(MOODPHONE[,2:26], as.numeric))
MOODPHONE$totaltime <- as.numeric(apply(MOODPHONE[,2:26], 1, sum))
names(MOODPHONE)[names(MOODPHONE) == "Online_Shopping"] <- "Shopping!"
MOODPHONE$Shopping <- (MOODPHONE$Shopping + MOODPHONE$`Shopping!`)
MOODPHONE <- MOODPHONE[,-c(19,21)]

MOODPHONE <- MOODPHONE[,c(1,40,2:39)]
MOODPHONE <- MOODPHONE[MOODPHONE[["totaltime"]] != 0.000]



## Feature selection for three models ------------------------------------------

MOODPHONE$Outcome_variable <- as.numeric(MOODPHONE$Outcome_variable)
#MOODPHONE <- MOODPHONE[MOODPHONE[["Outcome_variable"]] != 1,]
ONLYMOOD <- MOODPHONE[,c(1,26:40)]
ONLYPHONE <- MOODPHONE[,c(1:25,40)]


## CREATE THREE NEW CSV FILES FOR PYTHON ---------------------------------------

## Create a new file for the whole dataset--------------------------------------
write.csv(MOODPHONE, "MOODPHONE.csv", row.names = FALSE, 
          quote = FALSE)
## Create a new file for only mood dataset--------------------------------------
write.csv(ONLYMOOD , "ONLYMOOD.csv", row.names = FALSE, 
          quote = FALSE)

## Create a new file for only phone dataset-------------------------------------
write.csv(ONLYPHONE, "ONLYPHONE.csv", row.names = FALSE, 
          quote = FALSE)


