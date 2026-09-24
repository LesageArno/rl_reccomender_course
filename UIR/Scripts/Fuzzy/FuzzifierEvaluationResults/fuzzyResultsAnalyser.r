library(dplyr)
library(ggplot2)
library(plotly)
library(stringr)

# Function to retrieve the data files
collectData <- function(prefix = "on") {
    # From the working directory
    localPath <- getwd()
    
    # Detect all csv files beginning by the corresponding suffix
    l <- list.files(path=localPath, pattern = paste0(prefix, "*"))
    l <- l[endsWith(l, ".csv")]
    
    # Form all the paths to the csv, register source, extract them 
    lpath <- lapply(l, function(x) paste0(localPath, "/", x))
    df <- lapply(lpath, function(x) read.csv(x, stringsAsFactors = T))
    names(df) <- as.vector(l)
    
    # Return the different dataframe
    return (df)
}

# Function to plot taxonomy based fuzzification
plotTaxonomyBasedFuzzification <- function(dfx, confidence = 0.05, useRMSE = F, showPlot = T, returnDF = F) {
    # Extract the dataframe and summarise it
    df <- dfx %>% 
        
        # Group and summarise
        group_by(mode, p) %>% 
        summarise(
            mrmse = mean(RMSE),
            mnnacc = mean(NNACC),
            srmse = sd(RMSE),
            snnacc = sd(NNACC),
            n=n(),
            .groups = "drop"
        ) %>%
        
        # Error margin
        mutate(
            lbrmse = mrmse - qt(1-confidence/2, n-1)*srmse/sqrt(n),
            ubrmse = mrmse + qt(1-confidence/2, n-1)*srmse/sqrt(n),
            lbnnacc = mnnacc - qt(1-confidence/2, n-1)*snnacc/sqrt(n),
            ubnnacc = mnnacc + qt(1-confidence/2, n-1)*snnacc/sqrt(n)
            )
    
    # Plot with RMSE
    if (useRMSE && showPlot) {
        p <- ggplotly(
            ggplot(df, aes(x=p, y=mrmse, ymin=lbrmse, ymax=ubrmse)) +
                geom_ribbon(aes(fill=mode), alpha = 0.6) +
                geom_line(aes(colour = mode), show.legend = F, linewidth=1.2) +
                geom_line(aes(colour = mode, y=ubrmse), show.legend = F) +
                geom_line(aes(colour = mode, y=lbrmse), show.legend = F)
        )
    
    # Plot with NNACC
    } else if (showPlot) {
        p <- ggplotly(
            ggplot(df, aes(x=p, y=mnnacc, ymin=lbnnacc, ymax=ubnnacc)) +
                geom_ribbon(aes(fill=mode), alpha = 0.6) +
                geom_line(aes(colour = mode), show.legend = F, linewidth=1.2) +
                geom_line(aes(colour = mode, y=ubnnacc), show.legend = F) +
                geom_line(aes(colour = mode, y=lbnnacc), show.legend = F)
        )
    }
    
    # Force plot
    if (showPlot) print(p)
    if (returnDF) return(df)
}

# Function to plot taxonomy based fuzzification wrt gamma
plotGammaTaxonomyBasedFuzzification <- function(dfx, useRMSE = F, showPlot = T, returnDF = F) {
    # Extract the dataframe and summarise it
    df <- dfx %>% 
        
        # Group and summarise
        filter(mode %in% as.factor(c("weighted", "weightedLog2"))) %>%
        group_by(mode, p, gamma) %>% 
        summarise(
            mrmse = mean(RMSE),
            mnnacc = mean(NNACC),
            .groups = "drop"
        )
    
    # Plot with RMSE
    if (useRMSE && showPlot) {
        p <- ggplotly(
            ggplot(df, aes(x=p, y=gamma, fill=mrmse)) +
                geom_tile() +
                scale_fill_gradient2(low="darkgreen", mid="white",high="darkred", midpoint = 0.5) +
                facet_grid(~mode)
        )
        
        # Plot with NNACC
    } else if (showPlot) {
        p <- ggplotly(
            ggplot(df, aes(x=p, y=gamma, fill=mnnacc)) +
                geom_tile() +
                scale_fill_gradient2(low="darkred", mid="white",high="darkgreen", midpoint = 0.5) +
                facet_grid(~mode)
        )
    }
    
    # Force plot
    if (showPlot) print(p)
    if (returnDF) return(df)
}

# Function to plot rule based fuzzification wrt methods
plotGammaRuleBasedFuzzification <- function(dfx, confidence = 0.05, useRMSE = F, showPlot = T, returnDF = F) {
    # Extract the dataframe and summarise it
    df <- dfx %>% 
        
        # Group and summarise
        select(-subparam) %>%
        group_by(p, mode, method, threshold, code) %>%
        summarise(
            mrmse = mean(RMSE),
            mnnacc = mean(NNACC),
            srmse = sd(RMSE),
            snnacc = sd(NNACC),
            n = n(),
            .groups = "drop"
        ) %>%
        
        # Error margin
        mutate(
            lbrmse = mrmse - qt(1-confidence/2, n-1)*srmse/sqrt(n),
            ubrmse = mrmse + qt(1-confidence/2, n-1)*srmse/sqrt(n),
            lbnnacc = mnnacc - qt(1-confidence/2, n-1)*snnacc/sqrt(n),
            ubnnacc = mnnacc + qt(1-confidence/2, n-1)*snnacc/sqrt(n)
        )

    # Plot with RMSE
    if (useRMSE && showPlot) {
        p <- ggplotly(
            ggplot(df, aes(x=p, y=threshold, fill=mrmse)) +
                geom_tile() +
                scale_fill_gradient2(low="darkgreen", mid="white",high="darkred", midpoint = 0.5) +
                facet_grid(method~mode)
        )
        
        # Plot with NNACC
    } else if (showPlot) {
        p <- ggplotly(
            ggplot(df, aes(x=p, y=threshold, fill=mnnacc)) +
                geom_tile() +
                scale_fill_gradient2(low="darkred", mid="white",high="darkgreen", midpoint = 0.5) +
                facet_grid(method~mode)
        )
    }
    
    # Force plot
    if (showPlot) print(p)
    if (returnDF) return(df)
}

# Function to plot rule based fuzzification wrt best results among methods and mode
# use `plotGammaRuleBasedFuzzification` to have the df.
plotRuleBasedFuzzification <- function(df, bestAtp = 0.02, useRMSE = F, showPlot = T, returnDF = F) {
    # Extract the dataframe and summarise it

    if (useRMSE) selector <- df %>% filter(p==bestAtp) %>% group_by(method, mode, p) %>% filter(ubrmse == min(ubrmse))
    else selector <- df %>% filter(p==bestAtp) %>% group_by(method, mode, p) %>% filter(lbnnacc == max(lbnnacc))
    selector <- selector$code %>% droplevels()
    
    df <- df %>% filter(code %in% selector)
    
    # Plot with RMSE
    if (useRMSE && showPlot) {
        p <- ggplotly(
            ggplot(df, aes(x=p, y=mrmse, ymin=lbrmse, ymax=ubrmse)) +
                geom_ribbon(aes(fill = code), alpha = 0.6) +
                geom_line(aes(colour = code), show.legend = F, linewidth=1.2) +
                geom_line(aes(colour = code, y=ubrmse), show.legend = F) +
                geom_line(aes(colour = code, y=lbrmse), show.legend = F)
        )
        
        # Plot with NNACC
    } else if (showPlot) {
        p <- ggplotly(
            ggplot(df, aes(x=p, y=mnnacc, ymin=lbnnacc, ymax=ubnnacc)) +
                geom_ribbon(aes(fill = code), alpha = 0.6) +
                geom_line(aes(colour = code), show.legend = F, linewidth=1.2) +
                geom_line(aes(colour = code, y=ubnnacc), show.legend = F) +
                geom_line(aes(colour = code, y=lbnnacc), show.legend = F)
        )
    }
    
    # Force plot
    if (showPlot) print(p)
    if (returnDF) return(df)
}


# Function to add gamma as mode for taxonomy based method
encodeGammaAsMode <- function(dfx, gammaVal = 1, baseMode = "weightedLog2") {
    # Filter gamma and models
    dfa <- dfx %>%
        filter(gamma == gammaVal, mode == baseMode)

    # Function to rename the mode
    nameLog <- function(vec) {
        vec <- as.character(vec)
        vec[str_detect(vec, "Log2")] <- "Log"
        vec[str_detect(vec, "weighted")] <- ""
        return (vec)
    }
    
    # Rename and return modified dataframe
    dfa$mode <- as.factor(paste0("Gamma", nameLog(dfa$mode), dfa$gamma))
    return(bind_rows(dfa, dfx))
}

# Function to plot rules based association result, while returning best three combinations
plotAllRuleBased <- function(df, useRMSE = F, bestAtp = 0.02, selectHead = 3) {
    # Plot grid plot for rules association (k, p, RMSE/NNAC) and retrieve the df to put in for the next plot
    df_rules <- plotGammaRuleBasedFuzzification(df, useRMSE = useRMSE, returnDF = T)
    
    # Plot and get the best k per method for p=bestAtp
    df_rules <- plotRuleBasedFuzzification(df_rules, useRMSE = useRMSE, returnDF = T, bestAtp = bestAtp)
    
    # Get the best methods overall
    selector <- df_rules %>% group_by(mode, method, p) %>% filter(p==bestAtp) %>% arrange(desc(lbnnacc)) %>% head(selectHead)
    
    # Select all the informations relatives to the best methods
    selected <- df %>% filter(code%in%selector$code) %>% select(-mode, -method, -threshold, -subparam)
    
    # Rename weightedLog2 to Gamma 1
    selected$mode <- selected$code %>% str_replace("weightedLog2", "Gamma1")
    
    # Remove the code and return
    selected <- selected %>% select(-code)
    return(selected)
}

#### If __name__ == "__main__": ####
if (sys.nframe() == 0){
    ##### Collect the data ####
    df_original <- collectData()
    df <- df_original
    
    # Modify some dataframe
    ## Alt Taxonomy to include Gamma1 and Fixed
    df$onTaxonomyAltEvaluation.csv <- encodeGammaAsMode(bind_rows(
        df$onFixedAltEvaluation.csv, 
        df$onTaxonomyAltEvaluation.csv),
        gammaVal = 1, baseMode = "weighted"
    )
    
    ## Taxonomy to include Gamma1 and Fixed
    df$onTaxonomyEvaluation.csv <- encodeGammaAsMode(bind_rows(
        df$onFixedEvaluation.csv, 
        df$onTaxonomyEvaluation.csv),
        gammaVal = 1, baseMode = "weighted"
    )
    
    ## Add code to rule_based (Alt)
    df$onRulesAssociationsAltEvaluation.csv <- df$onRulesAssociationsAltEvaluation.csv %>%
        mutate(code=as.factor(paste0(mode,".",method,"_k",threshold)))
    
    ## Add code to rule_based
    df$onRulesAssociationsEvaluation.csv <- df$onRulesAssociationsEvaluation.csv %>%
        mutate(code=as.factor(paste0(mode,".",method,"_k",threshold)))
        
    #### Plots ####
    ##### Taxonomy #####
    # On (user) taxonomy evaluation (alternative)
    plotTaxonomyBasedFuzzification(df$onTaxonomyAltEvaluation.csv, useRMSE = T)
    plotGammaTaxonomyBasedFuzzification(df$onTaxonomyAltEvaluation.csv, useRMSE = F)
    
    # On (user) taxonomy evaluation <0.2, 0.5, 0.8>
    plotTaxonomyBasedFuzzification(df$onTaxonomyEvaluation.csv, useRMSE = T)
    plotGammaTaxonomyBasedFuzzification(df$onTaxonomyEvaluation.csv, useRMSE = F)
    
    ##### Association Rules #####
    ## On alternative
    selectedAltNNACC <- plotAllRuleBased(df$onRulesAssociationsAltEvaluation.csv, useRMSE = F, bestAtp = 0.02, selectHead = 50)
    selectedAltRMSE <- plotAllRuleBased(df$onRulesAssociationsAltEvaluation.csv, useRMSE = T, bestAtp = 0.02, selectHead = 50)
    
    ## On <0.2,0.5,0.8>
    selectedNNACC <- plotAllRuleBased(df$onRulesAssociationsEvaluation.csv, useRMSE = F, bestAtp = 0.02, selectHead = 50)
    selectedRMSE <- plotAllRuleBased(df$onRulesAssociationsEvaluation.csv, useRMSE = T, bestAtp = 0.02, selectHead = 50)
    
    ##### Global ####
    ## On user ALT Global
    df$onGlobalAltNNACC <- bind_rows(selectedAltNNACC, df$onTaxonomyAltEvaluation.csv)
    df$onGlobalAltRMSE <- bind_rows(selectedAltRMSE, df$onTaxonomyAltEvaluation.csv)
    
    plotTaxonomyBasedFuzzification(df$onGlobalAltNNACC, useRMSE = F) 
    plotTaxonomyBasedFuzzification(df$onGlobalAltRMSE, useRMSE = T)
    
    ## On user Global
    df$onGlobalNNACC <- bind_rows(selectedNNACC, df$onTaxonomyEvaluation.csv)
    df$onGlobalRMSE <- bind_rows(selectedRMSE, df$onTaxonomyEvaluation.csv)
    
    plotTaxonomyBasedFuzzification(df$onGlobalNNACC, useRMSE = F) 
    plotTaxonomyBasedFuzzification(df$onGlobalRMSE, useRMSE = T)
}
    
    