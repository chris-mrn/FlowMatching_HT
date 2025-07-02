import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

sns.set_context('paper', font_scale=3.5)
if __name__=="__main__":
    dfs = [2, 3]
    num_heavy = [1, 4]
    models = ["vanilla", "TAF", "gTAF", "mTAF(fix)"]
    seeds = [1, 2, 3]
    run = [1, 2, 3, 4, 5]

    # save all results in the following dfs:
    df_tst = pd.DataFrame(columns=num_heavy, index=models)
    df_tst_std = pd.DataFrame(columns=num_heavy, index=models)

    df_area_light = pd.DataFrame(columns=[str(heavy) + " light" for heavy in num_heavy], index=models)
    df_area_heavy = pd.DataFrame(columns=[str(heavy) + " heavy" for heavy in num_heavy], index=models)
    df_tvar_light = pd.DataFrame(columns=[str(heavy) + " light" for heavy in num_heavy], index=models)
    df_tvar_heavy = pd.DataFrame(columns=[str(heavy) + " heavy" for heavy in num_heavy], index=models)

    for df in dfs:
        for heavy in num_heavy:
            dict_wrong_light = {} # synth. tail estimates
            dict_wrong_heavy = {}
            if heavy==4 and df==2:
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
            else:
                fig, (ax1, ax2) = plt.subplots(1, 2)
            confusion = pd.DataFrame(index=["tn", "fn", "fp", "tp"], columns=models) # for synth. tail estimate
            for model in models:
                # test loss
                PATH_tst = f"df{df}h{heavy}/likelihood/nsf{model}5test.txt"
                file = open(PATH_tst)
                lines = file.readlines()
                mean_tst = np.mean(np.array([line.strip()[:-2] for line in lines], dtype=float))
                std_tst = np.std(np.array([line.strip()[:-2] for line in lines], dtype=float))
                df_tst[heavy][model] = mean_tst
                df_tst_std[heavy][model] = std_tst

                # area
                PATH_area = f"df{df}h{heavy}/area/nsf{model}.txt"
                file = open(PATH_area)
                lines = file.readlines()
                area = [line.strip() for line in lines]
                area_light = []
                area_heavy = []
                for line in area:
                    area_comp_wise = np.array([float(i) for i in line.split(" ")])
                    area_light.append(np.mean(area_comp_wise[:-heavy]))
                    area_heavy.append(np.mean(area_comp_wise[-heavy:]))

                df_area_light[str(heavy) + " light"][model]= np.mean(area_light)
                df_area_heavy[str(heavy) + " heavy"][model]= np.mean(area_heavy)

                # tvar
                PATH_tvar = f"df{df}h{heavy}/tvar/nsf{model}.txt"
                file = open(PATH_tvar)
                lines = file.readlines()
                tvar = [line.strip() for line in lines]
                tvar_light = []
                tvar_heavy = []
                iterate=True
                for line in tvar:
                    if iterate:
                        tvar_comp_wise = np.array([float(i) for i in line.split(" ")])
                        tvar_light.append(np.mean(tvar_comp_wise[:-heavy]))
                        tvar_heavy.append(np.mean(tvar_comp_wise[-heavy:]))
                        iterate=False
                    else:
                        iterate = True
                df_tvar_light[str(heavy) + " light"][model] = np.mean(tvar_light)
                df_tvar_heavy[str(heavy) + " heavy"][model] = np.mean(tvar_heavy)

                # synth tail_estimates
                PATH_tail_est = f"df{df}h{heavy}/synth_tailest/nsf{model}.txt"
                file = open(PATH_tail_est)
                lines = file.readlines()
                wrong_light = []
                wrong_heavy = []
                tail_estimates = [line.strip() for line in lines]
                for line in tail_estimates:
                    tail_est_compwise = np.array([float(i) for i in line.split(" ")])
                    # dfs above 10 are counted as light-tailed
                    for j in range(len(tail_est_compwise)):
                        if tail_est_compwise[j] >=10.0:
                            tail_est_compwise[j]=0

                    # light-tailed components
                    for comp in range(8 - heavy):
                        if tail_est_compwise[comp]>0.0:
                            wrong_light.append(tail_est_compwise[comp])
                    # heavy-tailed components:
                    for comp in range(8 - heavy, 8):
                        if tail_est_compwise[comp]==0.0:
                            wrong_heavy.append(10.)
                        else:
                            wrong_heavy.append(tail_est_compwise[comp])

                    dict_wrong_light[model] = wrong_light
                    dict_wrong_heavy[model] = wrong_heavy

                    # print confusion matrix:
                    # consider heavy-tailed as positives, light-tailed as negatives
                    positives = np.arange(8)[-heavy:]
                    negatives = np.arange(8)[:-heavy]
                    tp = 0
                    tn = 0
                    fp = 0
                    fn = 0
                    num_pos = 0
                    num_neg = 0
                    for line in tail_estimates:
                        tail_est_compwise = np.array([float(i) for i in line.split(" ")])
                        # dfs above 10 are counted as light-tailed
                        for j in range(len(tail_est_compwise)):
                            if tail_est_compwise[j] >= 10.0:
                                tail_est_compwise[j] = 0

                        # update true/false negatives, i.e. light-tailed components that are classified as light-tailed/heavy-tailed:
                        for comp in range(8 - heavy):
                            if tail_est_compwise[comp] > 0.0:
                                # predicted as heavy-tailed
                                fn += 1
                            else:
                                # predicted as light-tailed
                                tn += 1

                        # update true/false positives, i.e. heavy-tailed components that are classified as heavy-tailed/light-tailed:
                        for comp in range(8 - heavy, 8):
                            if tail_est_compwise[comp] > 0.0:
                                # predicted as heavy-tailed
                                tp += 1
                            else:
                                # predicted as light-tailed
                                fp += 1

                        num_pos += heavy
                        num_neg += 8 - heavy

                    confusion[model] = [tn, fn, fp, tp]

                    confusion_df = pd.DataFrame(np.array([[tn / (tn + fn), fp / (tp + fp)],
                                                          [fn / (fn + tn), tp / (tp + fp)]]),
                                                columns=["L", "H"],
                                                index=["L", "H"]
                                                )
                    plt.clf()
                    fig = plt.figure(figsize=(6, 6))
                    sns.heatmap(confusion_df, annot=True, cbar=False, fmt=".1%", annot_kws={"size": 45 / np.sqrt(len(confusion_df))})
                    plt.xlabel("Actual")
                    plt.ylabel("Generated")
                    plt.tight_layout()
                    # plt.matshow(confusion_mTAF, cols=["light-tailed", "heavy-tailed"], rows=["light-tailed", "heavy-tailed"])
                    fig.savefig(f"df{df}h{heavy}/synth_tailest/{model}.pdf")
            print(f"Degree of freedom={df}, Number of heavy-tailed components={heavy}")
            print(confusion )


            """
            ax1.yaxis.set_major_locator(MaxNLocator(integer=True))
            ax1.hist(list(dict_wrong_light.values()), label=["vanilla", "TAF", "gTAF", "mTAF(fix)"], bins=4)
            ax1.set_ylabel("Counts")
            ax1.set_xlabel("Estimated Tail Index")
            ax2.set_xlabel("Estimated Tail Index")
            ax2.hist(list(dict_wrong_heavy.values()), label=["vanilla", "TAF", "gTAF", "mTAF(fix)"], bins=4, align="right")
            if heavy==4 and df==2:
                ax2.legend(loc=9)
            elif heavy==1 and df==3:
                ax2.legend()
            plt.tight_layout()
            print(plt.rcParams.get('figure.figsize'))

            plt.savefig(f"plots/synth_tailest_df{df}h{heavy}.pdf")
            """
        # return results:
        print("###########################")
        print(f"## Degree of freedom = {df} ##")
        print("###########################")
        print("Log Likelihood Test loss")
        print("###########################")
        print("mean:")
        print(df_tst)
        print("std:")
        print(df_tst_std)
        print("")
        print("Area")
        print("##########################")
        print(df_area_light)
        print(df_area_heavy)
        print("")
        print("tVaR")
        print("##########################")
        print(df_tvar_light)
        print(df_tvar_heavy)
        print("")
