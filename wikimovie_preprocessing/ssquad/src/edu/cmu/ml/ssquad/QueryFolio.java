package edu.cmu.ml.ssquad;

import java.io.File;
import java.io.IOException;

public interface QueryFolio {

	void addDocumentSentences(File path) throws IOException;

	void addDocumentText(File path) throws IOException;

	String serialize(boolean delimit);

}