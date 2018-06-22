JAVAROOT = .

AM_JAVACFLAGS = -classpath "$$CLASSPATH":../$(PACKAGE)jni-$(VERSION).jar

jarfile = xflaim_edit.jar
jardir = $(datadir)/java
pkgpath = xedit

$(jarfile): $(dist_noinst_JAVA) 
	$(JAR) cvf $(JARFLAGS) $@ $(pkgpath)/*.class

jar_DATA = $(jarfile)

all-local: $(jarfile)

CLEANFILES = $(jarfile) $(pkgpath)/*.class

dist_noinst_JAVA = \
 $(pkgpath)/Attribute.java \
 $(pkgpath)/AttributeSelector.java \
 $(pkgpath)/Collection.java \
 $(pkgpath)/CollectionSelector.java \
 $(pkgpath)/Document.java \
 $(pkgpath)/DocumentSelector.java \
 $(pkgpath)/EditValueDialog.java \
 $(pkgpath)/FileImporter.java \
 $(pkgpath)/NodeDialog.java \
 $(pkgpath)/NodePanel.java \
 $(pkgpath)/NodeTag.java \
 $(pkgpath)/NodeType.java \
 $(pkgpath)/NodeValue.java \
 $(pkgpath)/OpenDocumentSelector.java \
 $(pkgpath)/ProgressBox.java \
 $(pkgpath)/Status.java \
 $(pkgpath)/UITools.java \
 $(pkgpath)/XEdit.java \
 $(pkgpath)/XFileFilter.java
