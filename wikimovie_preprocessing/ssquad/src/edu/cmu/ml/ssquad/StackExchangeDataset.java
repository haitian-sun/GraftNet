package edu.cmu.ml.ssquad;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.io.LineNumberReader;
import java.io.PrintWriter;
import java.util.Properties;
import java.util.Set;
import java.util.TreeSet;
import java.util.logging.Logger;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

import edu.cmu.ml.ssquad.store.DocumentStore;
import edu.cmu.ml.ssquad.store.DocumentStoreTron;

public class StackExchangeDataset {
	protected String STORE_PATH = "sentence-store";
	protected static final String SENT_PROP = "sentId";
	protected static final String THREAD_PROP = "threadId";
	protected static final Logger log = Logger.getLogger(StackExchangeDataset.class.getName());
	protected static final Pattern THREAD_ID = Pattern.compile("^([0-9]*)[qac].*");
	protected static final int MAX_LENGTH=2048;

	public StackExchangeDataset() {
		// TODO Auto-generated constructor stub
	}

	public static void main(String[] args) { 
		if (args.length < 3) {
			System.err.println("Usage:\n\t[-skip queryId] qaSentences.tsv cloze.tsv tagsByThread.tsv output.txt");
			System.err.println("If present, the skip queryId should be the last known good query processed.");
			System.exit(1);
		}
		int n=0;
		String skipto=null;
		if (args[0].equals("-skip")) {
			skipto=args[1];
			n=2;
		}
		try {
			new StackExchangeDataset().process(args[n], args[n+1], args[n+2], args[n+3], skipto); 
		} catch(Exception e) {
			e.printStackTrace();
			System.exit(1);
		}
	}
	
	protected String makeId(String[] lineFields) {
		return lineFields[0] + "@" + lineFields[1] + "@" + lineFields[3].indexOf("@placeholder");
	}

	public void process(String sentencesPath, String clozePath, String tagsPath, String outputFilename, String skipto) {
		File sentencesFile = new File(sentencesPath);
		File clozeFile = new File(clozePath);
		LineNumberReader cloze=null;
		File tagsFile = new File(tagsPath);
		StackExchangeCandidates candidates=null;
		File outputFile = new File(outputFilename);
		PrintWriter output=null;
		loadIndex(sentencesFile);
		long last,now; last=now=System.currentTimeMillis();
		try {
			cloze = new LineNumberReader(new FileReader(clozeFile));
			candidates = makeCandidateBuilder(tagsFile);
			output = new PrintWriter(new FileWriter(outputFile));
			String threadid="", qid="";
			Properties must=new Properties(), mustNot = new Properties();

			if (skipto != null) {
				boolean foundIt=false;
				log.info("Skipping to query "+skipto+"...");
				for(String line; (line=cloze.readLine()) != null;) {
					qid = makeId(line.split("\t", 4));
					if (qid.equals(skipto) && (foundIt=true) ) break; // side effect assignment
				}
				if (foundIt) log.info("Resuming processing at line "+cloze.getLineNumber()+" of "+clozePath);
				else throw new IllegalStateException("Couldn't find query with id '"+skipto+"'");
			}

			for(String line; (line=cloze.readLine()) != null; ) {
				String[] parts = line.split("\t",4);
				if (parts.length != 4) continue; //skip malformatted lines
				if (parts[3].length() > MAX_LENGTH) continue; // skip abnormally long "sentences"
				qid = makeId(parts);
				this.buildOtherFields(parseClozeId(parts), mustNot, must); // cheap hack

				final Set<String> queryCandidates = candidates.get(mustNot, must);
				Query query = new Query(qid, parts[3].trim(), parts[1], must, mustNot, DocumentStoreTron.DISK) {
					@Override
					public void buildCandidates() { this.candidates = queryCandidates; }
				};

				// do the lucene query
				query.build(forceIncludeAnswer());

				String ser = query.serialize(delimitDocuments());
				output.println(ser);
				output.println();
				output.println();
				//				System.out.println(ser);
				//				System.out.println();

				if ( (now=System.currentTimeMillis())-last > 2000) {
					last=now;
					log.info(cloze.getLineNumber()+" lines...");
				}
			}
		} catch (IOException e) {
			e.printStackTrace();
		} finally {
			try {
				cloze.close();
				candidates.finished();
				output.close();
			} catch (IOException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}
		}
	}

	protected boolean delimitDocuments() {
		return false;
	}

	protected boolean forceIncludeAnswer() {
		return true;
	}

	protected String[] parseClozeId(String[] fields) {
		return new String[]{"",fields[0]};
	}

	protected DocumentStore loadIndex(File sentencesFile) {
		boolean exists = new File(STORE_PATH).exists();
		DocumentStore store = DocumentStoreTron.DISK.get(STORE_PATH);
		if (exists) return store;
		LineNumberReader sentences=null;
		long last,now; last=now=System.currentTimeMillis();
		try {
			sentences = new LineNumberReader(new FileReader(sentencesFile));
			Properties otherFields = new Properties();
			for(String line; (line=sentences.readLine())!=null;) {
				String[] parts = line.split("\t",4); // split empty sentences for acounting purposes
				// skip everything but the actual sentences
				if (!parts[0].equals("sent")) continue;
				// expect 4 fields
				if (parts.length != 4) { 
					log.warning("malformatted line "+sentences.getLineNumber()+" of "+sentencesFile.getPath()+"; skipping");
					//throw new IllegalStateException("\n"+line);
					continue;
				}
				parts[3] = parts[3].trim();
				if (parts[3].length() == 0) continue; // skip empty sentences
				try {
					buildOtherFields(parts,otherFields,otherFields);
				} catch (IllegalStateException e) {
					System.err.println("Trouble at line "+sentences.getLineNumber()+":\n"+parts[1]+"\n"+line);
					sentences.close();
					throw e;
				}
				store.add(parts[3], otherFields.getProperty(SENT_PROP), otherFields);
				if ( (now=System.currentTimeMillis())-last > 2000) {
					last=now;
					log.info(sentences.getLineNumber()+" lines...");
				}
			}

			sentences.close();
			store.freeze();
			return store;
		} catch (IOException e) {
			e.printStackTrace();
		}
		return null;
	}

	protected void buildOtherFields(String[] fields, Properties sent, Properties thread) {
		String sentId=fields[1];
		sent.setProperty(SENT_PROP,sentId);
		Matcher m = THREAD_ID.matcher(sentId);//.group(1);
		m.find();
		String threadId = m.group(1); 
		thread.setProperty(THREAD_PROP, threadId);
	}

	protected StackExchangeCandidates makeCandidateBuilder(File data) throws FileNotFoundException {
		return new StackOverflowCandidatesByThread(data);
	}


	private class StackOverflowCandidatesByThread implements StackExchangeCandidates {
		LineNumberReader tags;
		String tagline=null,tagsThreadid=""; 
		Set<String> tagsForThread = null;

		public StackOverflowCandidatesByThread(File tagsFile) throws FileNotFoundException {
			tags = new LineNumberReader(new FileReader(tagsFile));
		}

		public Set<String> get(Properties mustNot, Properties must) throws IOException {

			String threadid=must.getProperty(StackExchangeDataset.THREAD_PROP);
			if (!tagsThreadid.equals(threadid)) {
				// then read next chunk
				//				log.info("Fetching candidates for thread "+threadid+"; searching from "+tagsThreadid);
				tagsForThread = new TreeSet<String>();
				if (tagline==null) tagline = tags.readLine();
				while( !tagline.startsWith(threadid) && (tagline=tags.readLine()) != null) {}
				if (tagline==null) throw new IllegalStateException("Ran out of thread tags while looking for thread "+threadid);
				tagsThreadid = tagline.substring(0, tagline.indexOf("\t"));
				do {
					int delim=tagline.trim().indexOf("\t");
					if (delim<1) continue;
					tagsForThread.add(tagline.substring(delim+1));
				} while( (tagline=tags.readLine()) != null && tagline.startsWith(threadid));
				//				log.info("Stopped at tagline=\n"+tagline);
			}
			return tagsForThread;
		}

		public void finished() throws IOException {
			tags.close();
		}


	}
}
