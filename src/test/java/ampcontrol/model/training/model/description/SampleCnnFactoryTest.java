package ampcontrol.model.training.model.description;

import org.junit.Test;

import java.util.ArrayList;

/**
 * Test cases for {@link SampleCnnFactory}
 *
 * @author Christian Skärby
 */
public class SampleCnnFactoryTest {

    /**
     * Just a smoke test...
     */
    @Test
    public void addModelData() {
        new FactoryInitializer()
                .setInputSize(new int[] {1, 4410})
                .initialize(
                        ((tr, ev, is, np, md) -> new SampleCnnFactory(tr,ev,is,np,md).addModelData(new ArrayList<>()))
                );
    }
}