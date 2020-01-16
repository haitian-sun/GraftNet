package edu.cmu.ml.ssquad.store;

public abstract class DocumentStoreTron {
	public static final DocumentStoreTron DEFAULT=new DocumentStoreTron() {
		@Override
		public DocumentStore get(String name) {
			return this.get();
		}
		public DocumentStore get() {
			return new MemoryDocumentStore();
		}
	};
	public static final DocumentStoreTron DISK=new DocumentStoreTron() {
		private DocumentStore store = null;
		public DocumentStore get() {return this.store;}
		public DocumentStore get(String path) {
			if (this.store == null) store = new DiskDocumentStore(path);
			return this.store;
		}
	};
	public abstract DocumentStore get(String name);
	public abstract DocumentStore get();
}
