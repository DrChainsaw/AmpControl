package ampcontrol.model.training.model.validation.listen;

import org.deeplearning4j.eval.Evaluation;

import java.io.IOException;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.function.Consumer;

/**
 * Check point for {@link Evaluation Evaluations}. Stores the result in a text file.
 */
public class EvalCheckPoint implements Consumer<Evaluation> {

    private final String fileName;
    private final String modelName;
    private final TextWriter.Factory writerFactory;

    /**
     * Constructor
     * @param fileName name of the file to write evaluation to
     * @param modelName Name of the model
     */
    public EvalCheckPoint(String fileName, String modelName, TextWriter.Factory writerFactory) {
        this.fileName = fileName;
        this.modelName = modelName;
        this.writerFactory = writerFactory;
    }

    @Override
    public void accept(Evaluation eval) {
        try {
            Path path = Paths.get(fileName);
            TextWriter writer = writerFactory.create(path);
            writer.write(modelName + "\n");
            writer.write(eval.stats());
            writer.close();

        } catch (IOException e) {
            e.printStackTrace();
        }
    }
}
