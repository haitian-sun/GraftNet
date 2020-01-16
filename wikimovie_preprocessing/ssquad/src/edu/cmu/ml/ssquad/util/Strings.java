package edu.cmu.ml.ssquad.util;

import java.io.StringReader;
import java.util.ArrayList;
import java.util.List;

import edu.stanford.nlp.ling.HasWord;
import edu.stanford.nlp.process.DocumentPreprocessor;

public class Strings {

	public static List<String> sentences(String s) {
		DocumentPreprocessor p = new DocumentPreprocessor(new StringReader(s));
		List<String> result = new ArrayList<String>();
		for (List<HasWord> sentence : p) {
			StringBuilder sb = new StringBuilder();
			for (HasWord tok : sentence) {
				sb.append(" ");
				sb.append(tok.word());
			}
			result.add(sb.substring(1));
		}
		return result;
	}
	
	public static String join(String delim, List<String> blocks) {
		StringBuilder sb = new StringBuilder();
		for (String i : blocks) sb.append(delim).append(i);
		return sb.substring(delim.length());
	}

}
