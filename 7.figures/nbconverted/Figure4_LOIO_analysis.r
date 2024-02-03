suppressPackageStartupMessages(library(dplyr))
suppressPackageStartupMessages(library(ggplot2))
suppressPackageStartupMessages(library(patchwork))

# Load variables important for plotting (e.g., themes, phenotypes, etc.)
source("themes.r")

# Set I/O
results_dir <- file.path("..", "3.evaluate_model", "evaluations", "LOIO_probas")

# Results from `get_LOIO_probabilities.ipynb`
results_file <- file.path(results_dir, "compiled_LOIO_probabilites.tsv")

# Results from `LOIO_evaluation.ipynb`
results_summary_file <- file.path(results_dir, "LOIO_summary_ranks_allfeaturespaces.tsv")
results_summary_perphenotype_file <- file.path(results_dir, "LOIO_summary_ranks_perphenotype_allfeaturespaces.tsv")

output_fig_loio <- file.path("figures", "main_figure_4_loio.png")

# Set high threshold which represents a more-or-less arbitrary p value cutoff
# for assigning high probability phenotypes to cells
high_threshold <- 0.9

# Set custom labellers for adding context to facet text plotting
custom_labeller <- function(value) {
  paste0("Correct prediction:\n", value)
}

shuffled_labeller <- function(value) {
  paste("Shuffled:", value)
}

# Create phenotype categories dataframe
# Note: phenotype_categories defined in themes.r
phenotype_categories_df <- stack(phenotype_categories) %>%
  rename(Mitocheck_Category = ind, Mitocheck_Phenotypic_Class = values)

phenotype_categories_df

loio_df <- readr::read_tsv(
    results_file,
    col_types = readr::cols(
        .default = "d",
        "Model_Phenotypic_Class" = "c",
        "Mitocheck_Phenotypic_Class" = "c",
        "Cell_UUID" = "c",
        "Metadata_DNA" = "c",
        "Model_Type" = "c",
        "Model_Balance_Type" = "c",
        "Model_Feature_Type" = "c"
    )
) %>%
    dplyr::select(!`...1`) %>%
    dplyr::group_by(
        Cell_UUID,
        Model_Type,
        Model_Balance_Type,
        Metadata_DNA,
        Mitocheck_Phenotypic_Class,
        Model_Feature_Type,
        Model_C,
        Model_l1_ratio
    ) %>%
    dplyr::mutate(rank_value = rank(desc(Predicted_Probability))) %>%
    dplyr::mutate(correct_pred = paste(Mitocheck_Phenotypic_Class == Model_Phenotypic_Class)) %>%
    dplyr::left_join(
        phenotype_categories_df,
        by = "Mitocheck_Phenotypic_Class"
    ) %>%
    dplyr::left_join(
        phenotype_categories_df,
        by = c("Model_Phenotypic_Class" = "Mitocheck_Phenotypic_Class"),
        suffix = c("", "_model")
    ) %>%
    dplyr::mutate(
        correct_class_pred = paste(Mitocheck_Category == Mitocheck_Category_model)
    )

loio_df$rank_value <- factor(loio_df$rank_value, levels = paste(sort(unique(loio_df$rank_value))))

# The `feature_spaces` variable is defined in themes.r
loio_df$Model_Feature_Type <-
    dplyr::recode_factor(loio_df$Model_Feature_Type, !!!feature_spaces)

refactor_logical <- c("TRUE" = "TRUE", "FALSE" = "FALSE")
loio_df$correct_pred <-
    dplyr::recode_factor(loio_df$correct_pred, !!!refactor_logical)

loio_df$Shuffled <- dplyr::recode_factor(
    loio_df$Model_Type,
    "final" = "FALSE", "shuffled_baseline" = "TRUE"
)

print(dim(loio_df))
head(loio_df, 5)

loio_feature_space_gg <- (
    ggplot(loio_df,
        aes(x = rank_value, y = Predicted_Probability)
          )
    + geom_boxplot(aes(fill = correct_pred), outlier.size = 0.1, lwd = 0.3)
    + theme_bw()
    + phenotypic_ggplot_theme
    + facet_grid(
        "Shuffled~Model_Feature_Type",
        labeller = labeller(Shuffled = shuffled_labeller)
    )
    + labs(x = "Rank of prediction", y = "Prediction probability")
    + scale_fill_manual(
        "Correct\nphenotype\nprediction?",
        values = focus_corr_colors,
        labels = focus_corr_labels
    )
)

loio_feature_space_gg

# Load per image, per phenotype, per feature space summary
loio_summary_per_phenotype_df <- readr::read_tsv(
    results_summary_file,
    col_types = readr::cols(
        .default = "d",
        "Metadata_DNA" = "c",
        "Model_Type" = "c",
        "Mitocheck_Phenotypic_Class" = "c",
        "Model_Balance_Type" = "c",
        "Model_Feature_Type" = "c"
    )
) %>%
    dplyr::mutate(loio_label = "Leave one image out") %>%
    # Generate a new column that we will use for plotting
    # Note, we define focus_phenotypes in themes.r
    dplyr::mutate(Mitocheck_Plot_Label = if_else(
        Mitocheck_Phenotypic_Class %in% focus_phenotypes,
        Mitocheck_Phenotypic_Class,
        "Other"
    )) %>%
    dplyr::filter(Model_Balance_Type == "balanced")

#feature_order <- c("CP" = "CP", "DP" = "DP", "CP_and_DP" = "CP_and_DP")

loio_summary_per_phenotype_df$Model_Feature_Type <-
    dplyr::recode_factor(loio_summary_per_phenotype_df$Model_Feature_Type, !!!feature_spaces)

loio_summary_per_phenotype_df$Mitocheck_Plot_Label <-
    dplyr::recode_factor(loio_summary_per_phenotype_df$Mitocheck_Plot_Label, !!!focus_phenotype_labels)

loio_summary_per_phenotype_df$Shuffled <- dplyr::recode_factor(
    loio_summary_per_phenotype_df$Model_Type,
    "final" = "FALSE", "shuffled_baseline" = "TRUE"
)

head(loio_summary_per_phenotype_df, 3)

per_image_category_gg <- (
    ggplot(loio_summary_per_phenotype_df, aes(x = Average_Rank, y = Average_P_Value))
    + geom_point(aes(size = Count, color = Shuffled), alpha = 0.2)
    + theme_bw()
    + phenotypic_ggplot_theme
    + facet_grid(
        "Mitocheck_Plot_Label~Model_Feature_Type"
    )
    + labs(
        x = "Average rank of correct label\n(per held out image)",
        y = "Average probability of correct label\n(per held out image)"
    )

    + scale_size_continuous(
        name = "Per held\nout image\ncell count"
    )
    + scale_color_manual(
        "Is data\nrandomly\nshuffled?",
        values = shuffled_colors
    )
    + geom_vline(xintercept=2, linetype = "dashed", color = "red")
    + theme(
        strip.text = element_text(size = 8.3),
    )
    + guides(
        color = guide_legend(
            order = 1,
            override.aes = list(size = 2, alpha = 1)
        ),
        linetype = guide_legend(
            order = 2,
            override.aes = list(alpha = 1)
        ),
    )
)

per_image_category_gg

phenotypic_class_category_counts <- loio_df %>%
    dplyr::ungroup() %>%
    dplyr::select(
        Mitocheck_Phenotypic_Class,
        Model_Type,
        Model_Balance_Type,
        Model_Feature_Type,
        correct_pred,
        correct_class_pred
    ) %>%
    dplyr::group_by(
        Mitocheck_Phenotypic_Class,
        Model_Type,
        Model_Balance_Type,
        Model_Feature_Type,
        correct_pred,
        correct_class_pred
    ) %>%
    dplyr::summarize(phenotype_count = n()) %>%
    dplyr::ungroup()

loio_thresh_df <- loio_df %>%
    dplyr::ungroup() %>%
    dplyr::mutate(pass_threshold = paste(Predicted_Probability >= high_threshold)) %>%
    dplyr::group_by(
        Mitocheck_Phenotypic_Class,
        Model_Type,
        Model_Balance_Type,
        Model_Feature_Type,
        correct_pred,
        correct_class_pred,
        pass_threshold
    ) %>%
    dplyr::summarize(count = n()) %>%
    dplyr::left_join(
        phenotypic_class_category_counts,
        by = c(
            "Mitocheck_Phenotypic_Class",
            "Model_Type",
            "Model_Balance_Type",
            "Model_Feature_Type",
            "correct_pred"
        )
    ) %>%
    dplyr::mutate(phenotype_prop = count / phenotype_count)

phenotypic_class_counts <- loio_df %>%
    dplyr::ungroup() %>%
    dplyr::select(
        Mitocheck_Phenotypic_Class,
        Model_Type,
        Model_Balance_Type,
        Model_Feature_Type,
        correct_pred
    ) %>%
    dplyr::group_by(
        Mitocheck_Phenotypic_Class,
        Model_Type,
        Model_Balance_Type,
        Model_Feature_Type,
        correct_pred
    ) %>%
    dplyr::summarize(phenotype_count = n()) %>%
    dplyr::ungroup()

loio_thresh_df <- loio_df %>%
    dplyr::mutate(pass_threshold = paste(Predicted_Probability >= high_threshold)) %>%
    dplyr::group_by(
        Mitocheck_Phenotypic_Class,
        Model_Type,
        Model_Balance_Type,
        Model_Feature_Type,
        correct_pred,
        pass_threshold
    ) %>%
    dplyr::summarize(count = n()) %>%
    dplyr::left_join(
        phenotypic_class_counts,
        by = c(
            "Mitocheck_Phenotypic_Class",
            "Model_Type",
            "Model_Balance_Type",
            "Model_Feature_Type",
            "correct_pred"
        )
    ) %>%
    dplyr::mutate(phenotype_prop = count / phenotype_count)

# Reverse order of predicted label for plotting
loio_thresh_df$Mitocheck_Phenotypic_Class <-
    factor(loio_thresh_df$Mitocheck_Phenotypic_Class, levels = rev(unique(loio_thresh_df$Mitocheck_Phenotypic_Class)))

loio_thresh_df$Shuffled <- dplyr::recode_factor(
    loio_thresh_df$Model_Type,
    "final" = "FALSE", "shuffled_baseline" = "TRUE"
)

head(loio_thresh_df)

correct_pred_proportion_gg <- (
    ggplot(
        loio_thresh_df %>%
            dplyr::filter(
                Model_Balance_Type == "balanced",
                Model_Feature_Type == "CellProfiler"
            ),
        aes(
            x = phenotype_prop,
            y = Mitocheck_Phenotypic_Class,
            fill = pass_threshold
        )
    )
    + geom_bar(stat = "identity")
    + geom_text(
        data = loio_thresh_df %>%
            dplyr::filter(
                Model_Balance_Type == "balanced",
                Model_Feature_Type == "CellProfiler",
                pass_threshold == TRUE
            ),
        color = "black",
        aes(label = count),
        nudge_x = 0.07,
        size = 3
    )
    + facet_grid(
        "Shuffled~correct_pred",
        labeller = labeller(correct_pred = custom_labeller, Shuffled = shuffled_labeller)
    )
    + theme_bw()
    + phenotypic_ggplot_theme
    + theme(axis.text = element_text(size = 7.5))
    + scale_fill_manual(
        paste0("Does cell\npass strict\nthreshold?\n(p = ", high_threshold, ")"),
        values = focus_corr_colors,
        labels = focus_corr_labels,
        breaks = c("TRUE", "FALSE")
    )
    + labs(x = "Cell proportions", y = "Mitocheck phenotypes")
)

correct_pred_proportion_gg

same_class_wrong_pred_summary_df <- loio_df %>%
    dplyr::filter(correct_pred == "FALSE") %>%
    dplyr::filter(rank_value == 1) %>%
    dplyr::group_by(
        Mitocheck_Phenotypic_Class,
        Model_Type,
        Model_Balance_Type,
        Model_Feature_Type,
        correct_class_pred
    ) %>%
    dplyr::summarize(count = n(), avg_prob = mean(Predicted_Probability)) %>%
    dplyr::ungroup() %>%
    dplyr::left_join(phenotype_categories_df, by = "Mitocheck_Phenotypic_Class") %>%
    dplyr::group_by(
        Mitocheck_Category,
        Model_Type,
        Model_Balance_Type,
        Model_Feature_Type,
        correct_class_pred
    ) %>%
    dplyr::summarize(total_count = sum(count), avg_prob = mean(avg_prob))

phenotypic_category_counts <- same_class_wrong_pred_summary_df %>%
    dplyr::select(
        Mitocheck_Category,
        Model_Type,
        Model_Balance_Type,
        Model_Feature_Type,
        total_count
    ) %>%
    dplyr::group_by(
        Mitocheck_Category,
        Model_Type,
        Model_Balance_Type,
        Model_Feature_Type
    ) %>%
    dplyr::summarize(phenotype_category_count = sum(total_count)) %>%
    dplyr::ungroup()

same_class_wrong_pred_summary_df <- same_class_wrong_pred_summary_df %>%
    dplyr::left_join(
        phenotypic_category_counts,
        by = c(
            "Mitocheck_Category",
            "Model_Type",
            "Model_Balance_Type",
            "Model_Feature_Type"
        )
    ) %>%
    dplyr::mutate(category_proportion = total_count / phenotype_category_count)

same_class_wrong_pred_summary_df$Mitocheck_Category <- factor(
    same_class_wrong_pred_summary_df$Mitocheck_Category,
    levels = rev(levels(same_class_wrong_pred_summary_df$Mitocheck_Category))
)

same_class_wrong_pred_summary_df$Shuffled <- dplyr::recode_factor(
    same_class_wrong_pred_summary_df$Model_Type,
    "final" = "FALSE", "shuffled_baseline" = "TRUE"
)

head(same_class_wrong_pred_summary_df)

correct_class_phenotype_pred_gg <- (
    ggplot(
        same_class_wrong_pred_summary_df %>%
            dplyr::filter(
                Model_Balance_Type == "balanced",
                Model_Feature_Type == "CellProfiler"
            ),
        aes(
            x = Mitocheck_Category,
            y = category_proportion,
            fill = correct_class_pred
        )
    )
    + geom_bar(stat = "identity")
    + facet_grid(
        "~Shuffled",
        labeller = labeller(Shuffled = shuffled_labeller)
    )
    + geom_text(
        data = same_class_wrong_pred_summary_df %>%
            dplyr::filter(
                correct_class_pred == TRUE,
                Model_Balance_Type == "balanced",
                Model_Feature_Type == "CellProfiler"
            ),
        color = "black",
        aes(label = total_count),
        nudge_y = 0.12
    )
    + coord_flip()
    + scale_fill_manual(
        "Correct\nphenotype\ncategory\nprediction?",
        values = focus_corr_colors,
        labels = focus_corr_labels,
        breaks = c("TRUE", "FALSE")
    )
    + theme_bw()
    + phenotypic_ggplot_theme
    + labs(x = "Mitocheck Pheno categories", y = "Cell proportions\nof incorrect phenotype predictions")
)

correct_class_phenotype_pred_gg

right_bottom_nested <- (
    correct_pred_proportion_gg / correct_class_phenotype_pred_gg
) + plot_layout(heights = c(1, 0.7)) 

bottom_nested <- (
    per_image_category_gg | right_bottom_nested
) + plot_layout(widths = c(1, 0.72))

compiled_fig <- (
    loio_feature_space_gg /
    bottom_nested
) + plot_annotation(tag_levels = "A") + plot_layout(heights = c(0.4, 1)) 

ggsave(output_fig_loio, dpi = 500, height = 11, width = 14)

compiled_fig