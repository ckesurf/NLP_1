
echo "--------------------------------------------"
echo "Now re-executing with custom improvements to HMM tagger for dealing low frequency words"
echo "First, we will try replacing all numerals with _NUMERAL_"

echo "First, replace words in ner_train.dat..."
python replace_numeral.py ner_train.dat
echo "Done.\n"

echo "Generating new ner.counts file using training data with _RARE_ and _NUMERAL_ replacements..."
python count_freqs.py ner_train.dat.replaced_numeral > ner.counts
echo "Done.\n"

echo "Tagging words in ner_dev.dat, output to prediction_file..."
python entity_tagger_numeral.py ner.counts ner_dev.dat > prediction_file
echo "Done.\n"

echo "Evaluating the prediction_file"
# evaluate the prediction_file, compare it to ner_dev.key
python eval_ne_tagger.py ner_dev.key prediction_file
echo "\n"

echo "Computing Viterbi algorithm on ner_dev.dat..."
python viterbi_numeral.py ner.counts ner_dev.dat > viterbi_predictions
echo "Done\n"

echo "Evaluating the viterbi_predictions"
python eval_ne_tagger.py ner_dev.key viterbi_predictions
echo "\n"