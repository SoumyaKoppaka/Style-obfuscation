#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --mem=12g
##SBATCH --nodelist=compute-0-7
#SBATCH -t 0

python src/main.py \
        --dataset gender \
        --clean_mem_every 5 \
        --reset_output_dir \
        --classifier_dir="pretrained_classifer/gender" \
        --train_src_file data/gender/blogs224n.train.src.txt \
        --train_trg_file data/gender/blogs224n.train.attr \
        --dev_src_file data/gender/blogs224n.dev.src.txt \
        --dev_trg_file data/gender/blogs224n.dev.attr \
        --dev_trg_ref data/gender/blogs224n.dev.src.txt \
        --src_vocab  data/gender/text.vocab \
        --trg_vocab  data/gender/attr.vocab \
        --d_word_vec=128 \
        --d_model=512 \
        --log_every=100 \
        --eval_every=1500 \
        --ppl_thresh=10000 \
        --batch_size 32 \
        --valid_batch_size 128 \
        --patience 5 \
        --lr_dec 0.5 \
        --lr 0.0008 \
        --dropout 0.3 \
        --max_len 10000 \
        --seed 0 \
        --beam_size 1 \
        --word_blank 0.2 \
        --word_dropout 0.1 \
        --word_shuffle 3 \
        --cuda \
        --anneal_epoch 5 \
        --temperature 0.01 \
        --klw 0.04 \
        --max_pool_k_size 1 \
        --bt \
        --lm \
        --gumbel_softmax \
        --avg_len \
        --no_styles 2 \
        --fl_len_control \
        --automated_multi_domain \
        --eval_bleu \
        --hard_len_stop \
        --reverse_len_control \
        #--vocab_boost \
        #--vocab_weights data/blogs_2dom/saved_dict.json \
        #--boost_w 20 \
        
        #      --strike_out_max \
