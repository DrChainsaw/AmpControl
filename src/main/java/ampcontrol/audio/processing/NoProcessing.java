package ampcontrol.audio.processing;

/**
 * No postprocessing. Output = input.
 *
 * @author Christian Skärby
 */
public class NoProcessing implements ProcessingResult.Factory {


    @Override
    public ProcessingResult create(ProcessingResult input) {
        return input;
    }

    @Override
    public String name() {
        return nameStatic();
    }

    public static String nameStatic() {
        return "nopp";
    }
}
