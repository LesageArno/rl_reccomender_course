library(dplyr)
library(ggplot2)
library(ggdist)
library(plotly)
library(boot)
library(readODS)

# Function to retrieve the data files
collectData <- function(prefix = "data") {
    # From the working directory
    localPath <- getwd()
    
    # Detect all csv files beginning by the corresponding suffix
    l <- list.files(path=localPath, pattern = paste0(prefix, "_*"))
    l <- l[endsWith(l, ".csv")]
    
    # Form all the paths to the csv, extract them and merge them
    l <- lapply(l, function(x) paste0(localPath, "/", x))
    df <- lapply(l, function(x) read.csv(x, stringsAsFactors = T))
    return (dplyr::bind_rows(df))
}

# https://www.geeksforgeeks.org/r-language/bootstrap-confidence-interval-with-r-programming/
boot.mean <- function(data, idx) {
    return(mean(data[idx], na.rm = T))
}

# Function to bootstrap a criterion average (note that 10k is maybe more suitable, but 1000 is the default in sns.lineplot [python])
lboot <- function(df, metric, criterion, n=1000) {
    i <- 1
    l <- list()
    print(metric)
    # For each method, k and iteration
    for (method in unique(df$Method)) {
        for (repetition in unique(df$k)) {
            for (iteration in unique(df$Iteration)) {
                # Select the rows possessing the specified method, metric, k and iteration over the defined criterion
                action <- as.vector(df %>% filter(Method==method, Metric==metric, k==repetition, Iteration==iteration) %>% select(as.symbol(criterion)))[[1]]
                
                # If no action was found (missing method for example), then continue
                if (length(action) == 0) {
                    l[[i]] <- list(Metric=metric, Method=method, k=repetition, Iteration=iteration, avg=NA, lower=NA, upper=NA)
                    i <- i+1 #Keep consistent size
                    next
                }
                
                # Compute bootstrap error margin at 95%, store the results and continue
                ci <- tryCatch({boot.ci(boot(action, boot.mean, R=n), type="perc")}, error = function(cond) {print(paste(method, metric, repetition, iteration, criterion))})
                l[[i]] <- list(Metric=metric, Method=method, k=repetition, Iteration=iteration, avg=mean(action), lower=ci$percent[4], upper=ci$percent[5])
                i <- i+1
                
                # Inform the user
                if (i%%250==0) {
                    print(i)   
                }
            }
        }
    }
    
    # Return the concatenated table
    df <- dplyr::bind_rows(l)
    df$id <- paste0(df$Metric, "_", df$Method)
    return(df)
}

# Function to plot the results
plotResults <- function(df, metric_vec, title = "avg", excludeMethod = c("")) {
    ggplotly(
        ggplot(
            df %>% filter(Metric %in% metric_vec, !(Method %in% excludeMethod)) %>% na.omit(),
            aes(x=Iteration, ymin=lower, ymax=upper, fill = Method, colour = Method)
        ) +
            geom_ribbon(aes(fill=Method), alpha=0.6) +
            geom_line(aes(y=avg), linewidth=1.2) +
            geom_line(aes(y=upper)) +
            geom_line(aes(y=lower)) +
            facet_grid(rows = vars(k), cols = vars(Metric), scales = "free_y") +
            ggtitle(title)
    )
}

plotResultsAsBoxPlot <- function(df, metric_vec, atIter=500000, title = "avg", excludeMethod = c("")) {
    ggplotly(
        ggplot(
            df %>% filter(Metric %in% metric_vec, Iteration==atIter, !(Method %in% excludeMethod)) %>% na.omit(),
            aes(x=Method, y = avg, ymin=lower, ymax=upper)
        ) +
            geom_pointrange(aes(colour = Method)) +
            geom_errorbar(aes(colour = Method), width = .5) +
            facet_grid(rows = vars(k), cols = vars(Metric), scales = "free_y") + 
            ggtitle(title) + 
            theme(axis.text.x = element_text(size = 7, angle = 20))
    )
}


# If __name__ == "__main__":
if (sys.nframe() == 0){
    # Collect the data
    df <- collectData()
    
    # Get for the average job
    avgJobDf <- dplyr::bind_rows(list(
    #    lboot(df, "UIR80", "Average.jobs", n=1000),
    #    lboot(df, "UIR100", "Average.jobs", n=1000),
    #    lboot(df, "altUIR80", "Average.jobs", n=1000),
    #    lboot(df, "altUIR100", "Average.jobs", n=1000),
    #    lboot(df, "altNNACCUIR80", "Average.jobs", n=1000),
        lboot(df, "altSplitNNACCUIR80", "Average.jobs", n=1000)
    ))
    
    # Get for the average reward
    #avgRewardDf <- dplyr::bind_rows(list(
    #    lboot(df, "UIR80", "Average.reward", n=1000),
    #    lboot(df, "UIR100", "Average.reward", n=1000),
    #    lboot(df, "altUIR80", "Average.reward", n=1000),
    #    lboot(df, "altUIR100", "Average.reward", n=1000),
    #    lboot(df, "altNNACCUIR80", "Average.reward", n=1000),
    #    lboot(df, "altSplitNNACCUIR80", "Average.reward", n=1000)
    #))
    
    # Get for the average preference coverage
    #avgPrefCoverageDf <- dplyr::bind_rows(list(
    #    lboot(df, "UIR80", "Average.pref.cov", n=1000),
    #    lboot(df, "UIR100", "Average.pref.cov", n=1000),
    #    lboot(df, "altUIR80", "Average.pref.cov", n=1000),
    #    lboot(df, "altUIR100", "Average.pref.cov", n=1000),
    #    lboot(df, "altNNACCUIR80", "Average.pref.cov", n=1000),
    #    lboot(df, "altSplitNNACCUIR80", "Average.pref.cov", n=1000)
    #))
     
    excludeMethods <- c("crisp", "minRuleFixedC2", "minRuleGamma1C2", "secondRuleFixedC2", "secondRuleGamma1C2", "weightedRuleFixedC2", "weightedRuleGamma1C2", "weightedRuleLogC2")
    
    # For UIR80
    plotResults(avgJobDf, c("altUIR80", "UIR80"), "Average Opened Job UIR80")
    plotResults(avgRewardDf, c("altUIR80", "UIR80"), "Average Reward UIR80")
    plotResults(avgPrefCoverageDf, c("altUIR80", "UIR80"), "Average Pref Coverage UIR80")
    
    plotResults(avgJobDf, c("altNNACCUIR80"), "Average Opened Job altNNACCUIR80")
    plotResults(avgRewardDf, c("altNNACCUIR80"), "Average Reward altNNACCUIR80")
    plotResults(avgPrefCoverageDf, c("altNNACCUIR80"), "Average Pref Coverage altNNACCUIR80")
    
    plotResults(avgJobDf, c("altSplitNNACCUIR80"), "Average Opened Job altSplitNNACCUIR80", excludeMethod = excludeMethods)
    plotResults(avgRewardDf, c("altSplitNNACCUIR80"), "Average Reward altSplitNNACCUIR80", excludeMethod = excludeMethods)
    plotResults(avgPrefCoverageDf, c("altSplitNNACCUIR80"), "Average Pref Coverage altSplitNNACCUIR80", excludeMethod = excludeMethods)
    
    ## Box plot
    plotResultsAsBoxPlot(avgJobDf, c("altSplitNNACCUIR80"), title = "Average Opened Job altSplitNNACCUIR80", excludeMethod = excludeMethods)
    #plotResultsAsBoxPlot(avgRewardDf, c("altSplitNNACCUIR80"), title = "Average Reward altSplitNNACCUIR80", excludeMethod = excludeMethods)
    #plotResultsAsBoxPlot(avgPrefCoverageDf, c("altSplitNNACCUIR80"), title = "Average Pref Coverage altSplitNNACCUIR80", excludeMethod = excludeMethods)
    
    # For UIR100
    plotResults(avgJobDf, c("altUIR100", "UIR100"), 2, "Average Opened Job UIR100")
    plotResults(avgRewardDf, c("altUIR100", "UIR100"), 2, "Average Reward Job UIR100")
    plotResults(avgPrefCoverageDf, c("altUIR100", "UIR100"), "Average Pref Coverage UIR100")
    
    # Export lboot results
    #readODS::write_ods(avgJobDf %>% filter(Iteration==500000, Metric=="altSplitNNACCUIR80") %>% na.omit(), path = "bootResults.ods", sheet = "avgJob")
    #readODS::write_ods(avgRewardDf %>% filter(Iteration==500000, Metric=="altSplitNNACCUIR80") %>% na.omit(), path = "bootResults.ods", sheet = "avgReward", append = T)
    #readODS::write_ods(avgPrefCoverageDf %>% filter(Iteration==500000, Metric=="altSplitNNACCUIR80") %>% na.omit(), path = "bootResults.ods", sheet = "avgPrefCov", append = T)
}

