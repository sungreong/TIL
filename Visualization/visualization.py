import numpy as np
import matplotlib.pyplot as plt
import itertools


def plot_confusion_matrix(cm = None , target_names=None, cmap=None,
                          normalize=True, labels=True, title='Confusion matrix' , criteria = "predicted" ):
    
    ## example 
    ## from sklearn.metrics import confusion_matrix
    #### cm = confusion_matrix(test_y, Rf_class)
    #### target_names = ["Not Rain", "Rain"]
    ###  criteria 기준을 어떻게 할지? 
    ##
    
    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy
    assert ( (criteria == "predicted") or (criteria == "true")  ),"predicted or true 로 선택해서 해주세요!"
    if criteria == "predicted" :
        cm = cm.astype('float') / cm.sum(axis=0)[np.newaxis, :]
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=0)[np.newaxis, :]
    elif criteria == "true" : 
        cm = cm.astype('float') / cm.sum(axis=1)[ : , np.newaxis]
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[: , np.newaxis]
    #cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    
        
    f, axes = plt.subplots(1, 1 , figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    
    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names)
        plt.yticks(tick_marks, target_names)
    
    if labels:
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            if normalize:
                plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                         horizontalalignment="center",
                         color="white" if cm[i, j] > thresh else "black")
            else:
                plt.text(j, i, "{:,}".format(cm[i, j]),
                         horizontalalignment="center",
                         color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
    plt.show()
    
    
"""
GAN KS Plot Total Loss 및 Loss 시각화
"""
from IPython.display import clear_output

def Check_KS(ks_output , Total_ks_graph , Margin , ROW , COL ) : 
    clear_output(wait= True)
    print("KS Plot")
    total = sess.run(G_sample , feed_dict=feed_dict_train)
    total = scaler.inverse_transform(total)
    g_plot = pd.DataFrame(total , columns = col )
    g_plot[fac_var] = g_plot[fac_var].round(0)
    Value = []
    for label in col :
        sample = g_plot[label]
        real   = x_plot[label]
        ks , p = stats.ks_2samp(real.values , sample.values)
        Value.append(ks)
    Total_KS = round( np.sum(Value) , 2)
    ks_2 = [iteration] + Value
    ks_3 = pd.DataFrame([ks_2], columns = ["iter"] + col)
    ks_output = ks_output.append(ks_3)
    fig , ax = plt.subplots(figsize=(26,13))
    fig.subplots_adjust(top = 0.95 , left = 0.03 , bottom = 0.05 , right = 0.99)
    updown = 0
    for name in col : 
        if updown % 2 == 0 :
            param , space="bottom" , "  "
        else : 
            param , space ="top" , "   "
        ax.plot(ks_output.iter , ks_output[[name]], label = name)
        ax.text(iteration , ks_output.loc[ks_output["iter"]==iteration , [name]].values , 
                space + name ,verticalalignment = param)
        updown +=1
    ax.set_title("KS [{}]".format(Total_KS) , fontsize = 30 )
    ax.set_xlabel("Epoch")
    ax.set_ylabel("KS")    
    box = ax.get_position()
    ax.set_position([box.x0, box.y0 + box.height * 0.1, box.width, box.height * 0.9])
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
              fancybox=True, shadow=True, ncol=15 , fontsize= 10)
    ax.text(iteration+1 , 0.05 , "   0.05", verticalalignment = param)
    ax.axhline(0.05, linewidth=4, color='r')
    ax.set_title("EPOCH : {} , KS : {}[{}][{}]".format(iteration-1 , 
                                                       Total_KS , 
                                                       Total_ks_graph.ks.min() ,len(col) ) , fontsize = 30)
    plt.savefig(model_dir +"/IND_KS_Log_{}.png".format(path_v))
    plt.show()
    """
    Total Graph
    """
    
    print("Total KS Plot")
    Total_ks_graph_ap = pd.DataFrame({"iter":[iteration] , "ks" :[Total_KS]})
    Total_ks_graph    = Total_ks_graph.append(Total_ks_graph_ap)
    #Total_ks_graph    = Total_ks_graph[Total_ks_graph.iter>0]
    fig , ax = plt.subplots(figsize = (26,13))
    fig.subplots_adjust(top = 0.95 , left = 0.03 , bottom = 0.04 , right = 0.99)
    ax.plot(Total_ks_graph.iter , Total_ks_graph.ks , linestyle ="-" , marker ="." , linewidth = 3, markersize = 12)
    ax.axhline(0.05, linewidth=4, color='r')
    ax.set_title("EPOCH : {} , KS : {}[{}][{}]".format(iteration-1 , Total_KS , Total_ks_graph.ks.min() ,len(col) ) , fontsize = 30)
    plt.savefig(model_dir +"/Total_KS_Log{}.png".format(path_v))
    plt.show()
    show_plot(row = ROW , ncol= COL )
    if Margin > Total_KS : 
        show_plot(row = ROW , ncol= COL )
        print("=======UPDate============")
        with open(model_dir + "/Change_Margin.txt", "a") as f:
            f.write("Epoch : {} , Margin : {} ===> {} \n".format(iteration , Margin , Total_KS))
        Margin = Total_KS
        ## 여기선 특정 샘플 저장해야 하므로. 원하는 개수 만큼.
        generate = {is_training_bn: True ,  use_moving_statistics: True}
        generate[z] = sample_z(50000, z_dim)
        total = sess.run(G_sample, feed_dict= generate )
        total = scaler.inverse_transform(total)
        g_plot = pd.DataFrame(total , columns = col )
        g_plot[fac_var] = g_plot[fac_var].round(0)
        g_plot.to_csv(model_dir + "/Generated_{}.csv".format(path_v) , index = False)
        
        saver.save(sess , model_dir + "/MODEL_{}".format(path_v))
    return ks_output , Total_ks_graph , Margin , Total_KS


from scipy import stats
def show_plot(row , ncol ) : 
    fig , axes = plt.subplots(row , ncol , figsize = (26,13))
    fig.subplots_adjust(hspace = 0.35 , wspace= 0.14 , top = 0.92 , left = 0.03 , bottom = 0.04 , right = 0.99)
    total = sess.run( G_sample , feed_dict=feed_dict_train )
    try : 
        total = total[~np.isnan(total).any(axis=1)]
        total = scaler.inverse_transform(total)
        g_plot = pd.DataFrame(total , columns = col )
        g_plot[fac_var] = g_plot[fac_var].round(0)
        """
        좀 더 쉬운 분포로 만들어서 학습시킨 후 다시 원래값으로 (factor 변수이기 때문에 가능하다 생각함.)
        """

        col2 = 0
        error = []
        for j in range(row) :
            for k in range(ncol) :
                try :
                    label = col[col2]
                    sample = g_plot.loc[: , label]
                    sample.name = "Gene"
                    real_0 = x_plot.loc[: , label]
                    real_0.name ="Real"
                    ks , p = stats.ks_2samp(real_0.values , sample.values)
                    error.append(ks)
                    col2 += 1
                    if label in fac_var : 
                        sns.distplot( sample , ax=axes[j , k], norm_hist =True , kde=False , hist_kws ={"color":"r" , "label" :"Gene", "rwidth":0.75})
                        sns.distplot(real_0 , ax=axes[j , k],norm_hist =True, kde=False , hist_kws ={"color":"g" , "label" :"Real", "rwidth":0.75})
                        axes[j , k].legend(fontsize = 10)
                    elif label in num_var : 
                        sns.distplot(  sample , ax=axes[j , k] ,
                                     kde_kws={"color": "r", "lw": 2, "label": "Gene" , "shade" : True} , hist =False , rug = False) #   
                        sns.distplot(  real_0 , ax=axes[j , k] ,
                                     kde_kws={"color": "g", "lw": 2, "label": "Real", "shade" : True } , hist =False , rug = False) # 
                        axes[j , k].legend(fontsize = 10 )
                    axes[j , k].set_title( label , loc ="center" , fontsize= 10 )
                    axes[j , k].set_xlabel(' ')
                except IndexError as e : 
                    axes[j , k].axis("off")
        
        KS_DIF = round(np.sum(error),2)
        plt.suptitle('EPOCH {} , D_loss : {} , G_loss : {} KS : {}'.format(i , dloss,gloss,KS_DIF) , fontsize= 30)
        #if KS_DIF < 9 : 
        plt.savefig(vis_dir +"/Vis_{}_{}.png".format(path_v,KS_DIF))
        #else : 
        #    plt.savefig(model_dir +"/Vis_{}.png".format(path_v))
        plt.show()
        

        fig , ax = plt.subplots(figsize = (26,13))
        fig.subplots_adjust(top = 0.95 , left = 0.03 , bottom = 0.04 , right = 0.99)

        ax.plot(output.iter , output.dloss , label ="dloss" , linestyle ="-" , marker ="." , linewidth = 4, markersize = 12)
        ax.plot(output.iter , output.gloss , label ="gloss" , linestyle ="-" , marker ="." , linewidth = 4, markersize = 12)
        ax.set_title("[{}] , EPOCH : {} , Dloss : {} , Gloss : {}".format(title , iteration-1 ,  dloss, gloss ), fontsize= 30)
        ax.set_ylim(-5, 15)
        box = ax.get_position()
        ax.set_position([box.x0, box.y0 + box.height * 0.1, box.width, box.height * 0.9])
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), fancybox=True, shadow=True, ncol=4 , fontsize= 20)
        plt.savefig(model_dir +"/LOSS_Log_{}.png".format(path_v))
        plt.show()
    
        return print("시각화")

    except Exception as e : 
        print(e)
