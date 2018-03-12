package ampControl.audio.processing;

import java.util.Collections;
import java.util.List;

/**
 * No postprocessing. Output = input.
 *
 * @author Christian Skärby
 */
public class NoProcessing implements ProcessingResult.Processing {

    private double[][] output;

    @Override
    public void receive(double[][] input) {
        output = input;
    }

    @Override
    public List<double[][]> get() {
        return Collections.singletonList(output);
    }

    @Override
    public String name() {
        return nameStatic();
    }

    public static String nameStatic() {
        return "nopp";
    }
}
