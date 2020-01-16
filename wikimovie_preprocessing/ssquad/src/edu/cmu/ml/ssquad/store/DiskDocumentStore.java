package edu.cmu.ml.ssquad.store;

import java.io.IOException;
import java.nio.file.Paths;
import java.util.logging.Logger;

import org.apache.lucene.store.SimpleFSDirectory;

import edu.cmu.ml.ssquad.StackExchangeDataset;

public class DiskDocumentStore extends DocumentStore {
	protected static final Logger log = Logger.getLogger(DocumentStore.class.getName());
	
	private String name;
	public DiskDocumentStore(String name) {
		this.name = name;
	}

	@Override
	protected void initDirectory() {
		try {
			this.idx = new SimpleFSDirectory(Paths.get(this.name));
			log.info("Opened lucene index at "+this.name);
		} catch (IOException e) {
			throw new IllegalStateException("Couldn't initialize lucene index at "+this.name, e);
		}
	}

}
