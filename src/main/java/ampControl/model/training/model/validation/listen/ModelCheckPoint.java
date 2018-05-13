package ampControl.model.training.model.validation.listen;

import org.deeplearning4j.nn.api.Model;
import org.deeplearning4j.util.ModelSerializer;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.io.IOException;

/**
 * Can serialize a {@link Model}. Swallows {@link IOException}.
 */
public class ModelCheckPoint {

    private static final Logger log = LoggerFactory.getLogger(ModelCheckPoint.class);

    private final String fileName;
    private final Model model;

    public ModelCheckPoint(String fileName, Model model) {
        this.fileName = fileName;
        this.model = model;
    }

    /**
     * Saves the model.
     */
    public void save() {
        try {
            log.info("Saving model: " + model);
            ModelSerializer.writeModel(model, new File(fileName), true);
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
}
