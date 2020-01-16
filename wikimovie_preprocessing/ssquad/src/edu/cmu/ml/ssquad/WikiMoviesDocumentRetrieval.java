package edu.cmu.ml.ssquad;

import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.io.FileNotFoundException;
import java.io.UnsupportedEncodingException;
import java.io.LineNumberReader;
import java.io.PrintWriter;
import java.io.BufferedReader;
import java.text.SimpleDateFormat;
import java.time.format.DateTimeFormatter;
import java.util.Date;
import java.util.logging.Logger;
import java.util.List;
import java.util.ArrayList;
import java.util.Properties;
import java.lang.Float;
import java.lang.Math;

import javax.print.attribute.standard.DateTimeAtCompleted;

import edu.cmu.ml.ssquad.store.*;

public class WikiMoviesDocumentRetrieval {
	private static final Logger log = Logger.getLogger(WikiMoviesDocumentRetrieval.class.getName());

	public WikiMoviesDocumentRetrieval() {
		// TODO Auto-generated constructor stub
	}

	public static void main(String[] args) { 
		if (args.length < 3) {
			System.err.println("Usage:\n\tquestions.txt documents.txt output.txt");
			System.exit(1);
		}
        
        // construct DocumentStore
        DocumentStore store = DocumentStoreTron.DEFAULT.get(); // for memory storage

        // add Documents from args[1]
        // each document is in the format: documentId<TAB>title<TAB>contents
        System.out.println("Adding documents to store ...");
        int n = 0;
        try (BufferedReader br = new BufferedReader(new FileReader(args[1]))) {
            String line;
            while ((line = br.readLine()) != null) {
                n++;
                if (n % 100000 == 0) System.out.println("At document " + String.valueOf(n));
                int delim = line.indexOf("\t");
                if (delim < 0) continue;
                String eID = line.substring(0, delim).trim();
                String titleDoc = line.substring(delim+1).trim();
                delim = titleDoc.indexOf("\t");
                if (delim < 0) continue;
                String title = titleDoc.substring(0, delim).trim();
                String doc = titleDoc.substring(delim+1).trim();
                Properties properties = new Properties();
                properties.put("title", title);
                store.add(doc, eID, properties, 1.0);
            }
        } catch (FileNotFoundException e) {
			e.printStackTrace();
        } catch (IOException e) {
			e.printStackTrace();
        }

        // run queries from args[0]
        ArrayList<String> additionalFields = new ArrayList<String>();
        additionalFields.add("title");
        try (PrintWriter writer = new PrintWriter(args[2], "UTF-8")) {
            try (BufferedReader br = new BufferedReader(new FileReader(args[0]))) {
                String line;
                while ((line = br.readLine()) != null) {
                    int delim = line.indexOf("\t");
                    if (delim < 0) continue;
                    String qID = line.substring(0, delim);
                    String question = line.substring(delim+1);
                    List<DocumentStore.SearchResult> results = store.search(question, null, null, additionalFields);
                    writer.print(qID + "\t");
                    for (DocumentStore.SearchResult r : results) {
                        writer.print(r.id + "=" + Float.toString(r.score) + ",");
                    }
                    writer.print("\n");
                }
            writer.close();
            } catch (FileNotFoundException e) {
                e.printStackTrace();
            } catch (IOException e) {
                e.printStackTrace();
            }
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        } catch (UnsupportedEncodingException e) {
            e.printStackTrace();
        }

    }
}
