package edu.cmu.ml.ssquad.store;

import java.io.IOException;
import java.io.StringReader;
import java.util.ArrayList;
import java.util.List;
import java.util.Properties;
import java.util.TreeSet;

import org.apache.lucene.analysis.standard.StandardAnalyzer;
import org.apache.lucene.document.Document;
import org.apache.lucene.document.StoredField;
import org.apache.lucene.document.TextField;
import org.apache.lucene.document.NumericDocValuesField;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.search.BooleanClause;
import org.apache.lucene.search.BooleanQuery;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.ScoreDoc;
import org.apache.lucene.search.TopDocs;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.RAMDirectory;
import org.apache.lucene.util.QueryBuilder;

//tx to:
//http://javatechniques.com/blog/lucene-in-memory-text-search-example/
//http://lucene.apache.org/core/6_2_1/demo/src-html/org/apache/lucene/demo/IndexFiles.html
//http://lucene.apache.org/core/6_2_1/demo/src-html/org/apache/lucene/demo/SearchFiles.html
public abstract class DocumentStore {

	private static final String TOPK_PROPERTY="ssquad.topk";
	private int TOPK;// = 20;
	protected Directory idx;
	protected IndexWriter writer;
	private IndexSearcher searcher = null;

	private void initialize() {
		TOPK = Integer.parseInt(System.getProperty(TOPK_PROPERTY, "20"));
		initDirectory();
		try {
			IndexWriterConfig conf = new IndexWriterConfig(new StandardAnalyzer());
			// Make an writer to create the index
			writer = new IndexWriter(idx,conf);
		}
		catch(IOException ioe) {
			ioe.printStackTrace();
		}

	}

	protected abstract void initDirectory();

	public void freeze() {
		if (this.idx == null) this.initialize();
		try {
			// Optimize and close the writer to finish building the index
			//			writer.optimize();
			writer.commit();
			writer.close();

			// Build an IndexSearcher using the in-memory index
			searcher = new IndexSearcher(DirectoryReader.open(idx));
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}

	public void add(String content) { this.add(content,null,null,1.0); }
	public void add(String content, String id, Properties otherFields) { this.add(content,id,otherFields,1.0); }
	public void add(String content, String id, Properties otherFields, double boost) {
		if (this.idx == null) this.initialize();
		if (this.searcher != null) throw new IllegalStateException("Tried to add document to frozen index");
		try {
			writer.addDocument(createDocument(content, id, otherFields, boost));
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}

	/**
	 * Make a Document object with an un-indexed title field and an
	 * indexed content field.
	 */
	private Document createDocument(String content, String id, Properties otherFields, double boost) {
		Document doc = new Document();
		//System.err.println("\n"+id);

		// Stored fields aren't indexed. Indexed fields aren't stored. SMH
		doc.add(new StoredField("content", content));
        TextField contentField = new TextField("text", new StringReader(content));
        contentField.setBoost(((float) boost));
		doc.add(contentField);
		if (id != null) doc.add(new StoredField("id",id));
		if (otherFields != null) {
			for (String field : otherFields.stringPropertyNames()) {
				//System.err.println(field+"\t"+otherFields.getProperty(field));
				doc.add(new TextField(field,new StringReader(otherFields.getProperty(field))));
			}
		}
		return doc;
	}

	public List<SearchResult> search(String text) throws IOException {
		return search(text, null, null, null);
	}
	public List<SearchResult> search(String text, Properties otherPosFields, Properties otherNegFields) throws IOException {
		return search(text, otherPosFields, otherNegFields, null);
	}
	/**
	 * Searches for the given string in the "content" field
	 */
	public List<SearchResult> search(String text, Properties otherPosFields, Properties otherNegFields, List<String> additionalFields) throws IOException {
		if (this.searcher == null) this.freeze();
		System.err.println(text);

		// Build a Query object
		QueryBuilder qb = new QueryBuilder(new StandardAnalyzer());
		TreeSet<String> pool = new TreeSet<String>();
		ArrayList<SearchResult> results = new ArrayList<SearchResult>();

		BooleanQuery.Builder bb = new BooleanQuery.Builder();
		for (String s : text.split("@placeholder")) {
			if (s == null || s.trim().length()<2) continue;
			org.apache.lucene.search.Query q = null;
			//System.err.println(s);
			try {
				q = qb.createPhraseQuery("text", s);
				if (q != null) bb.add(q,BooleanClause.Occur.SHOULD);
				q = qb.createBooleanQuery("text", s, BooleanClause.Occur.SHOULD);
				if (q != null) bb.add(q,BooleanClause.Occur.SHOULD);
                if (additionalFields != null) {
                    for (String field : additionalFields) {
                        q = qb.createPhraseQuery(field, s);
                        if (q != null) bb.add(q,BooleanClause.Occur.SHOULD);
                        q = qb.createBooleanQuery(field, s, BooleanClause.Occur.SHOULD);
                        if (q != null) bb.add(q,BooleanClause.Occur.SHOULD);
                    }
                }
			} catch(Exception e) {
				throw new IllegalStateException("while running query "+q+" for text '"+s+"'",e);
			}
		}
		if (otherPosFields != null) {
			for (String field : otherPosFields.stringPropertyNames()) {
				org.apache.lucene.search.Query q = qb.createPhraseQuery(field, otherPosFields.getProperty(field));
				if (q != null) bb.add(q,BooleanClause.Occur.MUST);
			}
		}
		if (otherNegFields != null) {
			for (String field : otherNegFields.stringPropertyNames()) {
				org.apache.lucene.search.Query q = qb.createPhraseQuery(field, otherNegFields.getProperty(field));
				if (q != null) bb.add(q,BooleanClause.Occur.MUST_NOT);
			}
		}

		//		org.apache.lucene.search.Query query = qb.createPhraseQuery("content", queryString);
		//		org.apache.lucene.search.Query query = qb.createBooleanQuery("text", queryString);
		BooleanQuery q = bb.build();
		//System.err.println(q.toString());
		// Search for the query
		TopDocs hits = searcher.search(q,TOPK*100);

		// add stored document contents
		if (hits.totalHits != 0 ) {
//			System.err.println(text);
			System.err.println(hits.totalHits+" results");
			for (ScoreDoc sd : hits.scoreDocs) {
				Document doc = searcher.doc(sd.doc);
				//System.err.println(sd.score+"\t"+sd.doc+"\t"+doc.get("content"));
				//System.err.println(sd.score+"\t"+sd.doc+"\t"+doc.get("id"));
                String id = doc.get("id");
				String content = doc.get("content");
				if (!pool.contains(content)) results.add(new SearchResult(sd.score,id,content));
				pool.add(content);
				if (results.size()==TOPK) break;
			}
		} else { System.err.println("No results"); }
		//results.addAll(pool);
		return results;
	}
	
	public static class SearchResult {
        public String id;
		public String content;
		public float score;
		public SearchResult(float sc, String id, String con) {
            this.id = id;
			this.score = sc;
			this.content = con;
		}
		public String serialize() {
			return this.score + "\t" + this.content;
		}
	}

}
