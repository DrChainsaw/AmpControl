package ampControl.model.training.model.validation.listen;

import org.deeplearning4j.eval.Evaluation;

import java.io.IOException;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.function.Consumer;

/**
 * Check point for {@link Evaluation Evaluations}. Stores the result in a text file.
 */
public class EvalCheckPoint implements Consumer<Evaluation> {

    private final String fileBaseName;
    private final String modelName;
    private final TextWriter.Factory writerFactory;

    /**
     * Constructor
     * @param fileBaseName base name of the file
     * @param modelName Name of the model
     */
    public EvalCheckPoint(String fileBaseName, String modelName, TextWriter.Factory writerFactory) {
        this.fileBaseName = fileBaseName;
        this.modelName = modelName;
        this.writerFactory = writerFactory;
    }

    @Override
    public void accept(Evaluation eval) {
        try {
            Path path = Paths.get(fileBaseName + ".score");
            TextWriter writer = writerFactory.create(path);
            writer.write(modelName + "\n");
            writer.write(eval.confusionToString());
            writer.write(eval.stats());
            writer.close();

        } catch (IOException e) {
            e.printStackTrace();
        }
    }
}
