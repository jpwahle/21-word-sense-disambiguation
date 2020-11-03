# usage: sh job_array.sh

for model in bert-base-uncased roberta-base distilbert-base-uncased albert-base-v2 facebook/bart-base xlnet-base-cased google/electra-base-discriminator; do
    qsub run_scc.sh $model
done
