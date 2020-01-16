package edu.cmu.ml.ssquad;

import java.io.File;
import java.io.IOException;
import java.util.List;
import java.util.Properties;
import java.util.Set;
import java.util.TreeSet;
import java.util.logging.Logger;

import edu.cmu.ml.ssquad.store.DocumentStore;
import edu.cmu.ml.ssquad.store.DocumentStoreTron;
import edu.cmu.ml.ssquad.store.MemoryDocumentStore;
import edu.cmu.ml.ssquad.util.Strings;
import edu.stanford.nlp.tagger.maxent.MaxentTagger;


public class Query implements QueryFolio {
	private static final String DEFAULT_MODEL_PATH = "models/english-left3words-distsim.tagger";
	private static final int TRUNCATE_SENTENCE_LENGTH=200;
	private static final Logger log = Logger.getLogger(Query.class.getName());
	private static MaxentTagger tagger;
	private DocumentStore store;
	public String id, question, answer;
	public List<DocumentStore.SearchResult> sentences;
	public Set<String> candidates;
	private Properties metaPos, metaNeg;
	private int ndocuments=0;
	public Query(String id, String question, String answer) {
		this(id,question,answer,null,null,DocumentStoreTron.DEFAULT);
	}
	public Query(String id, String question, String answer, Properties metaPos, Properties metaNeg, DocumentStoreTron tron) {
		this.id = id;
		this.question = Strings.join("", Strings.sentences(question));
		this.answer = answer.replace("*CORRECT*","").toLowerCase();
		this.candidates = new TreeSet<String>();
		this.store = tron.get();
		this.metaPos = metaPos;
		this.metaNeg = metaNeg;
	}
	private static MaxentTagger getTagger() {
		if (tagger == null) {
			tagger = new MaxentTagger(System.getProperty("ssquad.taggerModel", DEFAULT_MODEL_PATH));
		}
		return tagger;
	}

	/* (non-Javadoc)
	 * @see edu.cmu.ml.ssquad.QueryFolio#addDocumentSentences(java.io.File)
	 */
	@Override
	public void addDocumentSentences(File path) throws IOException {
		// push though ProcessHTML
		// add sentences to store
		ProcessHTML p = new ProcessHTML();
		int n = 0;
		for (String sentence : p.sentences(path)) {
			store.add(sentence.substring(0,Math.min(sentence.length(), TRUNCATE_SENTENCE_LENGTH)));
			n++;
		}
		ndocuments++;
		//System.out.println("Added "+n+" sentences for query "+id);
	}
	
	/* (non-Javadoc)
	 * @see edu.cmu.ml.ssquad.QueryFolio#addDocumentText(java.io.File)
	 */
	@Override
	public void addDocumentText(File path) throws IOException {
		// push though ProcessHTML
		// add sentences to store
		ProcessHTML p = new ProcessHTML();
		int n = 0;
		StringBuilder sb= new StringBuilder();
		for (String sentence : p.sentences(path,2048)) {
			sb.append(" ").append(sentence);
			n++;
		}
		store.add(n>0?sb.substring(1):"");
		ndocuments++;
		//System.out.println("Added "+n+" sentences for query "+id);
	}

	public boolean build() throws IOException { return build(true); }
	public boolean build(boolean forceAnswer) throws IOException {
		if (ndocuments>0) log.info(ndocuments+" documents for query "+id);
		sentences = store.search(question,metaPos,metaNeg);//, answer);
		buildCandidates();
		if (forceAnswer) {
			candidates.add(answer+"\t-1\t-1");
		}
		return this.sentences.size()>0;
	}

	public void buildCandidates() {
		if (!this.candidates.isEmpty()) return;
		for (int i=0; i<sentences.size(); i++) {
			buildCandidate(getTagger(), sentences.get(i).content, i);
		}
	}
	public void buildCandidate(MaxentTagger tagger, String sent, int sentid) {
		String tagged = tagger.tagTokenizedString(sent);
		StringBuilder sb = null;
		String workingTag = null;
		int startTokenId=-1;
		int i=-1;
		for (String t : tagged.split(" ")) { i++;
			int delim = t.lastIndexOf("_");
			if (delim < 0) {
				throw new IllegalStateException("Invalid input from tagger: '"+t+"' in '"+tagged+"'");
			}
			String tok = t.substring(0,delim);
			String tag = t.substring(delim+1);
			if (sb != null) {
				// then we already have an np in progress
				if (tag.equals(workingTag)) { 
					sb.append(" ").append(tok); 
					// skip to the next token
					continue; 
				} else { 
					candidates.add(sb.append("\t").append(sentid).append("\t").append(startTokenId).toString().trim().toLowerCase()); sb = null;
				}
			} 
			// should we start a new NP?
			if (tag.startsWith("NN")) { sb = new StringBuilder(tok); workingTag = tag; startTokenId=i; }
		}
		// finish the final NP
		if (sb != null) candidates.add(sb.append("\t").append(sentid).append("\t").append(startTokenId).toString().trim().toLowerCase());
	}

	/* (non-Javadoc)
	 * @see edu.cmu.ml.ssquad.QueryFolio#serialize(boolean)
	 */
	@Override
	public String serialize(boolean delimit) {
		StringBuilder sb = new StringBuilder(id);
		if (delimit) {
			sb.append("\n");
			for(DocumentStore.SearchResult sent : sentences) sb.append("\n").append(sent.serialize());
		} else {
			sb.append("\n\n");
			for(DocumentStore.SearchResult sent : sentences) sb.append(sent.serialize()).append(" ");
			
		}
		sb.append("\n\n");
		sb.append(question);
		sb.append("\n\n");
		sb.append(answer);
		sb.append("\n");
		for(String cand : candidates) {
			if (cand.length() == 0) continue;
			sb.append("\n").append(cand);
		}
		return sb.toString();
	}
}
