package guru.springframework.springairagexpert.bootstrap;

import guru.springframework.springairagexpert.config.VectorStoreProperties;
import io.milvus.client.MilvusServiceClient;
import io.milvus.grpc.DataType;
import io.milvus.param.IndexType;
import io.milvus.param.MetricType;
import io.milvus.param.collection.*;
import io.milvus.param.highlevel.collection.ListCollectionsParam;
import io.milvus.param.highlevel.collection.response.ListCollectionsResponse;
import io.milvus.param.index.CreateIndexParam;
import lombok.extern.slf4j.Slf4j;
import org.springframework.ai.document.Document;
import org.springframework.ai.reader.tika.TikaDocumentReader;
import org.springframework.ai.transformer.splitter.TextSplitter;
import org.springframework.ai.transformer.splitter.TokenTextSplitter;
import org.springframework.ai.vectorstore.VectorStore;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.CommandLineRunner;
import org.springframework.stereotype.Component;

import java.util.List;


@Slf4j
@Component
public class LoadVectorStore implements CommandLineRunner {

    @Autowired
    VectorStore vectorStore;

    @Autowired
    VectorStoreProperties vectorStoreProperties;

    @Autowired
    MilvusServiceClient milvusServiceClient;

    @Override
    public void run(String... args) throws Exception {

        boolean response = milvusServiceClient.hasCollection(
                HasCollectionParam.newBuilder().withCollectionName("vector_store").build()
        ).getData();

        ListCollectionsParam listParam = ListCollectionsParam.newBuilder()
                .build();

        ListCollectionsResponse responseList = milvusServiceClient.listCollections(listParam).getData();

        List<String> collections = responseList.collectionNames;
        for(String s : collections) {
            log.info("Collection names :: {}", s);
        }

        milvusServiceClient.dropCollection(
                DropCollectionParam.newBuilder()
                        .withCollectionName("vector_store")
                        .build()
        );


    if (true) {
        // Create or load vectors
        //log.info("No collection found");
            log.info("Collection does not exist → creating...");
            CreateCollectionParam createCollectionParam = CreateCollectionParam.newBuilder()
                    .withCollectionName("vector_store")
                    .withDescription("My vector store collection")
                    .withShardsNum(2)
                    .addFieldType(
                            FieldType.newBuilder()
                                    .withName("id")
                                    .withDescription("Primary key")
                                    .withDataType(DataType.Int64)
                                    .withPrimaryKey(true)
                                    .withAutoID(true)
                                    .build()
                    )
                    .addFieldType(
                            FieldType.newBuilder()
                                    .withName("embedding")
                                    .withDescription("Vector embedding")
                                    .withDataType(DataType.FloatVector)
                                    .withDimension(1536) // use your embedding dimension!
                                    .build()
                    )
                    .addFieldType(
                            FieldType.newBuilder()
                                    .withName("doc_id")
                                    .withDescription("Document ID or chunk ID")
                                    .withDataType(DataType.VarChar)
                                    .withMaxLength(256)  // adjust as needed
                                    .build()
                    )
                    .addFieldType(
                            FieldType.newBuilder()
                                    .withName("content")
                                    .withDescription("Text chunk content")
                                    .withDataType(DataType.VarChar)
                                    .withMaxLength(65535)
                                    .build()
                    )
                    .addFieldType(
                            FieldType.newBuilder()
                                    .withName("metadata")
                                    .withDescription("Any extra metadata")
                                    .withDataType(DataType.VarChar)
                                    .withMaxLength(1024)  // adjust as needed
                                    .build()
                    )
                    .build();

             milvusServiceClient.createCollection(createCollectionParam);

        // 3. Create index (optional but recommended)
        milvusServiceClient.createIndex(
                CreateIndexParam.newBuilder()
                        .withCollectionName("vector_store")
                        .withFieldName("embedding")
                        .withIndexName("idx_embedding")
                        .withIndexType(IndexType.IVF_FLAT)  // or IVF_SQ8 etc.
                        .withMetricType(MetricType.L2)
                        .withExtraParam("{\"nlist\":1024}")
                        .build()
        );
        System.out.println("Index created!");

// 4. Load collection
        milvusServiceClient.loadCollection(
                LoadCollectionParam.newBuilder()
                        .withCollectionName("vector_store")
                        .build()
        );
             Thread.sleep(5000L);
            log.info("Collection created: vector_store");
    }


        log.info("Vector store name :: {}", vectorStore.getName());
        if(vectorStore.similaritySearch("Sportsman").isEmpty()) {
            log.info("Loading documents into vector store");
            vectorStoreProperties.getDocumentsToLoad().forEach(document -> {
                log.info("Loading doucment :: "+document.getFilename());
                TikaDocumentReader documentReader = new TikaDocumentReader(document);
                List<Document> documents = documentReader.get();
                TextSplitter textSpiltter = new TokenTextSplitter();
                List<Document> documentListSplit = textSpiltter.apply(documents);
                vectorStore.add(documentListSplit);
            });
        }
        log.debug("Vector store loaded ...");
    }
}
