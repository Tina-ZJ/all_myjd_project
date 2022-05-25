import re
import sys

reload(sys)
sys.setdefaultencoding('utf8')


def transform_ner_2_line_format(infile_path, outfile_path):
    labels =set()
    with open(outfile_path, "wb") as outfile:
        with open(infile_path, "r") as infile:
            for line in infile:
                line = re.split("[\t ]+", line.strip().decode("utf-8"))
                sb = []
                for token in line:
                    token = token.split("/")
                    word = token[0]
                    word_len = len(word)
                    if len(token) == 1:
                        print line
                        sb.append(word[0] + "/O")
                        for i in range(1, word_len - 1):
                            sb.append(word[i] + "/O")
                    else:
                        label = token[1]
                        labels.add(label)
                        if word_len == 1:
                            sb.append(word + "/S_" + label)
                        else:
                            sb.append(word[0] + "/B_" + label)
                            for i in range(1, word_len - 1):
                                sb.append(word[i] + "/I_" + label)
                            sb.append(word[word_len - 1] + "/E_" + label)
                outfile.write("\t".join(sb) + "\n")
        outfile.flush()
        print ",".join(labels)


if __name__ == "__main__":
    transform_ner_2_line_format("1.txt","best_data_6w_person_corpus_three_line.txt")
