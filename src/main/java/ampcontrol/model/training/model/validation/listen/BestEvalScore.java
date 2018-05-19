package ampcontrol.model.training.model.validation.listen;

import org.deeplearning4j.eval.Evaluation;

import java.io.BufferedReader;
import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.function.Consumer;
import java.util.function.Supplier;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * Keeps track of the best eval score of a model. Will check at a given location to test if
 * a model exits already and read the score from there.
 *
 * @author Christian Skärby
 */
public class BestEvalScore implements Consumer<Evaluation>, Supplier<Double> {

    private final static Pattern accPattern = Pattern.compile("\\s*Accuracy:\\s*(\\d*,?\\d*)");

    private double bestAccuracy = 0;

    /**
     * Constructor
     * @param fileName file name for accuracy data
     * @throws IOException if file can not be opened
     */
    public BestEvalScore(String fileName) throws IOException {
        final File modelFile = new File(fileName);
        if (modelFile.exists()) {
            //TODO: Move this method into this class instead
            bestAccuracy = getAccuracy(modelFile.getAbsolutePath());
        }
    }

    @Override
    public void accept(Evaluation evaluation) {
        bestAccuracy = Math.max(bestAccuracy, evaluation.accuracy());
    }

    @Override
    public Double get() {
        return bestAccuracy;
    }

    public static double getAccuracy(String path) throws IOException {
        BufferedReader scoreReader = Files.newBufferedReader(Paths.get(path));

        return scoreReader.lines()
                .map(accPattern::matcher)
                .filter(Matcher::matches)
                .map(matcher -> matcher.group(1))
                .map(str -> str.replace(",", "."))
                .mapToDouble(Double::parseDouble)
                .findFirst()
                .orElse(0);
    }
}
