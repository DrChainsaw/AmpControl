package ampControl.audio.processing;

import java.util.ArrayList;
import java.util.List;

/**
 * Connects two {@link Processing} so that output from the first is input to the second
 *
 * @author Christian Skärby
 */
public class Pipe implements ProcessingResult.Processing {

    final Processing first;
    final ProcessingResult.Processing second;

    private final List<double[][]> result = new ArrayList<>();

    public Pipe(ProcessingResult.Processing first, ProcessingResult.Processing second) {
        this.first = first;
        this.second = second;
    }

    @Override
    public void receive(double[][] input) {
        result.clear();
        first.receive(input);
        first.get().forEach(midRes -> {
            second.receive(midRes);
            result.addAll(second.get());
        });
    }

    @Override
    public String name() {
        return first.name() + nameStatic() + second.name();
    }

    public static String nameStatic() {
        return "_pipe_";
    }

    @Override
    public List<double[][]> get() {
        return result;
    }
}
