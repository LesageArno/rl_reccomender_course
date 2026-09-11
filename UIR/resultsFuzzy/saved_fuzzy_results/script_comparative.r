library(dplyr)
library(ggplot2)
library(ggdist)
library(plotly)
library(boot)

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
    return(mean(data[idx]))
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
                
                # Compute bootstrap error margin at 95%, store the results and continue
                ci <- boot.ci(boot(action, boot.mean, R=n), type="perc")
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

plotResults <- function(df, metric_vec, k_, title = "avg") {
    ggplotly(
        ggplot(
            avgJobDf %>% filter(Metric %in% metric_vec),
            aes(x=Iteration, ymin=lower, ymax=upper)
        ) +
            geom_ribbon(aes(fill=Method), alpha=0.6) +
            geom_line(aes(y=avg, colour=Method), linewidth=1.2) +
            geom_line(aes(y=upper, colour=Method)) +
            geom_line(aes(y=lower, colour=Method)) +
            facet_grid(rows = vars(k), cols = vars(Metric), scales = "free_y") +
            ggtitle(title)
    )
}


# If __name__ == "__main__":
if (sys.nframe() == 0){
    # Collect the data
    df <- collectData()
    
    # Get for the average job
    avgJobDf <- dplyr::bind_rows(list(
        lboot(df, "UIR80", "Average.jobs", n=1000),
        lboot(df, "UIR100", "Average.jobs", n=1000),
        lboot(df, "altUIR80", "Average.jobs", n=1000),
        lboot(df, "altUIR100", "Average.jobs", n=1000)
    ))
    
    # Get for the average reward
    avgRewardDf <- dplyr::bind_rows(list(
        lboot(df, "UIR80", "Average.reward", n=1000),
        lboot(df, "UIR100", "Average.reward", n=1000),
        lboot(df, "altUIR80", "Average.reward", n=1000),
        lboot(df, "altUIR100", "Average.reward", n=1000)
    ))
     
    # For Opened Job
    plotResults(avgJobDf, c("altUIR80", "UIR80"), 2, "Average Opened Job UIR80")
    plotResults(avgRewardDf, c("altUIR80", "UIR80"), 2, "Average Reward UIR80")
    
    # For Reward
    plotResults(avgJobDf, c("altUIR100", "UIR100"), 2, "Average Opened Job UIR100")
    plotResults(avgRewardDf, c("altUIR100", "UIR100"), 2, "Average Reward Job UIR100")
}

