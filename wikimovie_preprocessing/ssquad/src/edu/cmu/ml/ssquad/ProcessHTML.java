package edu.cmu.ml.ssquad;

import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.StringReader;
import java.net.MalformedURLException;
import java.net.URL;
import java.util.ArrayList;
import java.util.List;

import edu.cmu.ml.ssquad.util.Strings;
import edu.stanford.nlp.ling.HasWord;
import edu.stanford.nlp.process.DocumentPreprocessor;
import net.htmlparser.jericho.CharacterReference;
import net.htmlparser.jericho.Element;
import net.htmlparser.jericho.HTMLElementName;
import net.htmlparser.jericho.Source;
import net.htmlparser.jericho.StartTag;

public class ProcessHTML {

	public ProcessHTML() {
		// TODO Auto-generated constructor stub
	}

        public List<String> sentences(File path) throws IOException {
	    return sentences(path,-1);
	}
    public List<String> sentences(File path, int lengthLimit) throws IOException {
		InputStream in = new FileInputStream(path);
		Source source=new Source(in);
		
		// Call fullSequentialParse manually as most of the source will be parsed.
		source.fullSequentialParse();
		String text=source.getTextExtractor().setIncludeAttributes(false).toString().replaceAll("\\p{C}", "");
		if (lengthLimit>0) text = text.substring(0,Math.min(lengthLimit,text.length()));
		return Strings.sentences( text );
	}

	public static void main(String[] args) throws MalformedURLException, IOException {
		// input: HTML file
		// output: plaintext
		// output: sentences
		// 
		if (args.length < 1) {
			System.err.println("Usage:\n\t[infile.html] > output.txt");
			System.exit(1);
		}
		
		ProcessHTML p = new ProcessHTML();

		for (String sentence : p.sentences(new File(args[0]))) {
			System.out.println(sentence);
		}
//		System.out.println("\nSame again but this time extend the TextExtractor class to also exclude text from P elements and any elements with class=\"control\":\n");
//		TextExtractor textExtractor=new TextExtractor(source) {
//			public boolean excludeElement(StartTag startTag) {
//				return startTag.getName()==HTMLElementName.P || "control".equalsIgnoreCase(startTag.getAttributeValue("class"));
//			}
//		};
//		System.out.println(textExtractor.setIncludeAttributes(true).toString());
  }

	private static String getTitle(Source source) {
		Element titleElement=source.getFirstElement(HTMLElementName.TITLE);
		if (titleElement==null) return null;
		// TITLE element never contains other tags so just decode it collapsing whitespace:
		return CharacterReference.decodeCollapseWhiteSpace(titleElement.getContent());
	}

	private static String getMetaValue(Source source, String key) {
		for (int pos=0; pos<source.length();) {
			StartTag startTag=source.getNextStartTag(pos,"name",key,false);
			if (startTag==null) return null;
			if (startTag.getName()==HTMLElementName.META)
				return startTag.getAttributeValue("content"); // Attribute values are automatically decoded
			pos=startTag.getEnd();
		}
		return null;
	}
}
