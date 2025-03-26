from CASER import predict,predict_subtype
import ML_models as ml
import DL_models as dl

'''
Input file template
    Gene       VEST_score_missense  VEST_score_frameshift_indels  ...  label（is_caner_related)
    A1CF   0.4211              0.0059                        ...   1
    AAAS   0.3861              0.0347                        ...   0
    AAK1   0.2987              0.0136                        ...   0
Ouput file template
    Gene	Score	    p-value	            q-value                 is_cancer_related(<=threshold)
    ZNF217	0.76280177	0.0001692126179831	0.009896958830103773    1
    SATB2	0.7569754	0.0002654619970193	0.009896958830103773    1
    ...
    CSNK2A3	0.2800701	0.984669957774466	0.9886165307515179      0
    HSPA1A	0.2791872	0.9897121833084948	0.9916916076751118      0
    TSSK6	0.25791362	0.9951704545454544	0.9951704545454544      0
'''
# Gene       VEST_score_missense  VEST_score_frameshift_indels  ...  label（is_caner_related)
# A1CF   0.4211              0.0059                        ...   1
# AAAS   0.3861              0.0347                        ...   0
# AAK1   0.2987              0.0136                        ...   0
known_gene_path = './data/geneSet.txt'
unknown_gene_path = './data/unknonwn_gene.txt'
res_path = './result/final_result.txt'
predict(known_gene_path,unknown_gene_path,res_path)

'''
    Subtype prediction(Input multiple files and output multiple files)
'''
# known_gene_path = ['./data/cERs_gene.txt',
#                    './data/readers_gene.txt','./data/remodelers_gene.txt',
#                    './data/erasers_gene.txt','./data/writers_gene.txt']
# unknown_gene_path = ['./data/unknonwn_cERs_gene.txt',
#                    './data/unknonwn_readers_gene.txt','./data/unknonwn_remodelers_gene.txt',
#                    './data/unknonwn_erasers_gene.txt','./data/unknonwn_writers_gene.txt']
# res_path = ['./data/cERs_final_result.txt',
#                './data/readers_final_result.txt','./data/remodelers_final_result.txt',
#                './data/erasers_final_result.txt','./data/writers_final_result.txt']
# predict_subtype(known_gene_path,unknown_gene_path,res_path)

'''
    Select another model  
    deep semi-supervised methods include TemporalEnsembling,FlexMatch,VAT,MixMatch,LadderNetwork,UDA
    machine semi-supervised methods include SemiBoost,Tri_Training,Co_Training,LapSVM,Assemble,TSVM,SSGMM
    machine supervised methods include GDBT,Xgboost,LogisticRegression,RandomForestClassifier,SVC
'''
# known_gene_path = './data/geneSet.txt'
# unknown_gene_path = './data/unknonwn_gene.txt'
# res_path = './result/final_result.txt'
# model = ml.tri_training
# #model = dl.dl.flexmatch
# predict(known_gene_path,unknown_gene_path,res_path,model=model)