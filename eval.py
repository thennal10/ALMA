import argparse
import json
import os
import xml.etree.ElementTree as ET
from lxml import etree
from difflib import SequenceMatcher
from comet import download_model, load_from_checkpoint

FUZZY_THRESHOLD = 0.9
def parse_arguments():
    parser = argparse.ArgumentParser(description="Evaluate XML tag translation fidelity.")
    parser.add_argument('--reference-dir', type=str, required=True, 
                        help="Path to reference JSON files.")
    parser.add_argument('--pairs', type=str, required=True, 
                        help="Comma-separated language pairs (e.g., 'en-de,en-fr').")
    parser.add_argument('--hypothesis-dir', type=str, required=True, 
                        help="Path to plain text hypothesis files.")
    return parser.parse_args()

def fuzzy_match(a, b, threshold=0.9):
    return SequenceMatcher(None, a, b).ratio() >= threshold

def extract_tags_and_contents(xml_text):
    """Parses XML safely and returns all (tag, inner_text) tuples recursively."""
    try:
        # Wrap in root in case top-level multiple tags
        wrapped = f"<root>{xml_text}</root>"
        root = ET.fromstring(wrapped)
    except ET.ParseError:
        return []

    result = []

    def recurse(elem):
        # Get all text under this element
        inner_text = "".join(elem.itertext()).strip()
        if elem.tag != "root":  # Skip wrapper
            result.append((elem.tag, inner_text))
        for child in elem:
            recurse(child)

    recurse(root)
    return result

def evaluate_tags(reference_texts, hypothesis_texts, threshold=0.9):
    total_ref_tags = 0
    total_matches = 0

    sentence_wise_matches = []
    for ref, hyp in zip(reference_texts, hypothesis_texts):
        ref_tags = extract_tags_and_contents(ref)
        hyp_tags = extract_tags_and_contents(hyp)
        
        matched = [False] * len(hyp_tags)
        for tag, ref_content in ref_tags:
            total_ref_tags += 1
            for i, (h_tag, h_content) in enumerate(hyp_tags):
                if not matched[i] and tag == h_tag and fuzzy_match(ref_content, h_content, threshold):
                    total_matches += 1
                    matched[i] = True
                    break
        sentence_wise_matches.append({
            "reference": ref,
            "hypothesis": hyp,
            "matchs": matched,
        })

    total_pred_tags = sum(len(extract_tags_and_contents(h)) for h in hypothesis_texts)
    
    precision = total_matches / total_pred_tags if total_pred_tags > 0 else 0
    recall = total_matches / total_ref_tags if total_ref_tags > 0 else 0
    f1 = (2 * precision * recall) / (precision + recall) if precision + recall > 0 else 0

    return precision, recall, f1, sentence_wise_matches

def main():
    args = parse_arguments()
    output_dir = args.hypothesis_dir
    for lang_pair in args.pairs.split(','):
        ref_path = os.path.join(args.reference_dir, lang_pair, f"test.{lang_pair}.json")
        hyp_path = os.path.join(args.hypothesis_dir, f"test-{lang_pair}")
        
        source_lang, target_lang = lang_pair.split('-')
        references = []
        sources = []
        with open(ref_path, 'r', encoding='utf-8') as ref_file:
            for line in ref_file:
                data = json.loads(line)
                references.append(data['translation'][target_lang])
                sources.append(data['translation'][source_lang])
        
        with open(hyp_path, 'r', encoding='utf-8') as hyp_file:
            hypotheses = [line.strip() for line in hyp_file.readlines()]
        
        if len(references) != len(hypotheses):
            raise ValueError(f"Mismatch in number of references and hypotheses for {lang_pair}")

        
        # --- XML tag evaluation ---
        precision, recall, f1, tag_matches = evaluate_tags(references, hypotheses, FUZZY_THRESHOLD)
        
        # save tag matches to a file
        with open(f"{output_dir}/test.{lang_pair}.tag_matches", 'w', encoding='utf-8') as f:
            f.write("reference\thypothesis\tmatchs\n")
            for match in tag_matches:
                f.write(f"{match['reference']}\t{match['hypothesis']}\t{','.join(map(str, match['matchs']))}\n")
            # write summary
            f.write(f"f1: {f1}\n")
        
        print(f"[{lang_pair}] XML Tag Translation Evaluation:")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall:    {recall:.4f}")
        print(f"  F1 Score:  {f1:.4f}")

        # --- Comet evaluation --- 
        
        def remove_tags(text):
            try:
                tree = etree.fromstring(f"<root>{text}</root>")
                notags = etree.tostring(tree, encoding='utf-8', method='text').decode('utf-8')
                return notags.strip()
            except etree.XMLSyntaxError:
                print(f"Error parsing XML: {text}")
                return text.strip()
        
        comet_data = [
            {
                "src": remove_tags(src),
                "ref": remove_tags(ref),
                "mt": remove_tags(hyp)
            }
            for src, ref, hyp in zip(sources, references, hypotheses)
        ]
        
        model_path = download_model("Unbabel/wmt22-comet-da")
        model = load_from_checkpoint(model_path)
        
        scores = model.predict(comet_data, batch_size=32, gpus=1)
        
        print(f"[{lang_pair}] COMET Score: {scores.system_score:.4f}")
        with open(f"{output_dir}/test.{lang_pair}.comet", 'w', encoding='utf-8') as f:
            f.write("reference\thypothesis\tscore\n")
            for score, ref, hyp in zip(scores.scores, references, hypotheses):
                f.write(f"{ref}\t{hyp}\t{score}\n")
            f.write(f"system_score: {scores.system_score}\n")
        

if __name__ == "__main__":
    main()
