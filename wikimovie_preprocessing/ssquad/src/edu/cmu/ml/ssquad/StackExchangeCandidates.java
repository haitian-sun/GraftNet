package edu.cmu.ml.ssquad;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.io.LineNumberReader;
import java.util.Properties;
import java.util.Set;
import java.util.TreeSet;

public interface StackExchangeCandidates {
	public Set<String> get(Properties mustNot, Properties must) throws IOException;
	public void finished() throws IOException;
}
