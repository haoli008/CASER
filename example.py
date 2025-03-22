from CASER import predict
known_gene_path = './data/geneSet.txt'
unknown_gene_path = './data/unknonwn_gene.txt'
res_path = './result/final_result.txt'
predict(known_gene_path,unknown_gene_path,res_path)