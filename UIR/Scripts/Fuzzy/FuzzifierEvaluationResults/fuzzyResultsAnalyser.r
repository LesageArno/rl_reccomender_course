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
    plotTaxonomyBasedFuzzification(df$onTaxonomyAltEvaluation.csv, useRMSE = F)
    plotGammaTaxonomyBasedFuzzification(df$onTaxonomyAltEvaluation.csv, useRMSE = F)
    
    ##### Association Rules #####
    ## Alt NNACC
    df_rulesNNACC <- plotGammaRuleBasedFuzzification(df$onRulesAssociationsAltEvaluation.csv, useRMSE = F, returnDF = T)
    df_rulesNNACC <- plotRuleBasedFuzzification(df_rulesNNACC, useRMSE = F, returnDF = T, bestAtp = 0.02)
    selectorNNACC <- df_rulesNNACC %>% group_by(mode, method, p) %>% filter(p==0.02) %>% arrange(desc(lbnnacc)) %>% head(3)
    selectedNNACC <- df$onRulesAssociationsAltEvaluation.csv %>% 
        filter(code%in%selectorNNACC$code) %>%
        select(-mode, -method, -threshold, -subparam)
    selectedNNACC$mode <- selectedNNACC$code %>% str_replace("weightedLog2", "Gamma1")
    selectedNNACC <- selectedNNACC %>% select(-code)
    
    ## Alt RMSE
    df_rulesRMSE <- plotGammaRuleBasedFuzzification(df$onRulesAssociationsAltEvaluation.csv, useRMSE = T, returnDF = T)
    df_rulesRMSE <- plotRuleBasedFuzzification(df_rulesRMSE, useRMSE = T, returnDF = T, bestAtp = 0.02)
    selectorRMSE <- df_rulesRMSE %>% group_by(mode, method, p) %>% filter(p==0.02) %>% arrange(ubrmse) %>% head(3)
    selectedRMSE <- df$onRulesAssociationsAltEvaluation.csv %>% 
        filter(code%in%selectorRMSE$code) %>%
        select(-mode, -method, -threshold, -subparam)
    selectedRMSE$mode <- selectedRMSE$code %>% str_replace("weightedLog2", "Gamma1")
    selectedRMSE <- selectedRMSE %>% select(-code)
    
    ##### Global ####
    ## On user alt NNACC global
    df$onGlobalNNACC <- bind_rows(selectedNNACC, df$onTaxonomyAltEvaluation.csv)
    plotTaxonomyBasedFuzzification(df$onGlobalNNACC)
    
    ## On user alt RMSE global
    df$onGlobalRMSE <- bind_rows(selectedRMSE, df$onTaxonomyAltEvaluation.csv)
    plotTaxonomyBasedFuzzification(df$onGlobalRMSE, useRMSE = T)
}
    
    