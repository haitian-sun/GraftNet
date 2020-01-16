function strip( thing ) {
    sub("  *$","",thing);sub("^  *","",thing);
    return thing;
}
function finish( id,doc,question,answer,candidates,    A,n,i ) {
    answer=strip(answer);
    doc=strip(doc);
    if (index(answer," ")==0 && length(doc)>0) {
	printf "%s\n",id > "accepted.ids.txt"
	print id; print doc; print question; print answer;
	n=split(candidates,A);
	candidates = "";
	for (i=1;i<=n;i++) {
	    A[i]=strip(A[i]);
	    if (index(A[i]," ")==0) { candidates=candidates FS A[i]; }
	}
	print substr(candidates,length(FS)+1);
	return 0;
    }
    printf "%s\n",id > "rejected.ids.txt"
}
BEGIN { RS=ORS="\n\n";FS=OFS="\n";state="id"; }
{
    if (state == "id") {
	if (id) { finish(id,doc,question,answer,candidates); }
	id=$0; state="doc";
    }
    else if (state == "doc") { doc=$0; state="question"; }
    else if (state == "question") { question=$0; state="answer"; }
    else if (state == "answer") { answer=$0; state="candidates"; }
    else if (state == "candidates") { candidates=$0; state="id"; }
}
END { finish(id,doc,question,answer,candidates); }
