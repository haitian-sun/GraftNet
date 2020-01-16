package edu.cmu.ml.ssquad.store;

import org.apache.lucene.store.RAMDirectory;

public class MemoryDocumentStore extends DocumentStore {

	@Override
	protected void initDirectory() {
		// Construct a RAMDirectory to hold the in-memory representation
		// of the index.
		this.idx = new RAMDirectory();
	}

}
