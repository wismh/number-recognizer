#pragma once

#include "help/help_topic_table.h"
#include "help/help_view_model.h"
#include "view/digit_canvas.h"
#include "view/grid_row.h"
#include "view/influence_paint.h"
#include "view/list_item.h"
#include "view/neuron_row.h"
#include "view/picture_paint.h"
#include "view/probability_paint.h"
#include "view/prototype_paint.h"

#include <engine/ui/view_model.h>

#include <array>
#include <memory>
#include <string>

namespace game {

// Everything recognizer.xml binds. Behavior lives in Recognizer and DatasetManager; this type only holds state and the
// paints, plus the tab switch, which is state of the screen itself.
class RecognizerViewModel final : public engine::ui::ViewModel {
public:
    RecognizerViewModel();

    // Window-wide.
    engine::ui::Bindable<std::string> statusText{"Starting..."};
    engine::ui::Bindable<bool> tabRecognize{true};
    engine::ui::Bindable<bool> tabNeurons{false};
    engine::ui::Bindable<bool> tabTrain{false};
    engine::ui::Bindable<bool> tabDatasets{false};
    engine::ui::Bindable<bool> tabHelp{false};
    engine::ui::MethodCommand showRecognize;
    engine::ui::MethodCommand showNeurons;
    engine::ui::MethodCommand showTrain;
    engine::ui::MethodCommand showDatasets;
    engine::ui::MethodCommand showHelp;
    // Language buttons: the commands are bound by the Game, which owns the catalog.
    engine::ui::Bindable<bool> localeIsEn{true};
    engine::ui::Bindable<bool> localeIsUk{false};
    engine::ui::MethodCommand localeEn;
    engine::ui::MethodCommand localeUk;

    // Recognize tab.
    engine::ui::Bindable<std::string> predictionText{"Prediction: -"};
    engine::ui::Bindable<bool> modelLoaded{false};
    engine::ui::Bindable<bool> modelMissing{true};
    engine::ui::Bindable<std::string> modelNameText{"Model: none"};
    engine::ui::Bindable<std::string> modelArchText{"Architecture: -"};
    engine::ui::Bindable<std::string> modelLayersText{"Layers: -"};
    engine::ui::Bindable<std::string> modelEpochsText{"Epochs trained: -"};
    engine::ui::Bindable<std::string> testText{"MNIST test: -"};
    engine::ui::Bindable<std::string> testMutatedText{"MNIST test, mutated: -"};
    engine::ui::Bindable<std::string> hardestDigitText{"Most errors: -"};
    engine::ui::Bindable<std::string> easiestDigitText{"Fewest errors: -"};
    engine::ui::Bindable<std::string> perDigitText{"Accuracy per digit: -"};
    engine::ui::Bindable<std::string> addLabelText{"0"};
    engine::ui::Bindable<std::string> addTargetText{"Add to: MNIST (read-only)"};
    engine::ui::MethodCommand clearCanvas;
    engine::ui::MethodCommand addLabelUp;
    engine::ui::MethodCommand addLabelDown;
    engine::ui::MethodCommand addSample;
    DigitCanvas canvasPaint;
    ProbabilityPaint probabilityPaint;
    PrototypePaint prototypePaint;

    // Neurons tab. The Weights and Dream buttons are drawn twice, as the current mode and as the other one.
    engine::ui::Bindable<std::string> neuronsLayerText;
    engine::ui::Bindable<std::string> neuronsProgressText;
    engine::ui::Bindable<std::string> neuronTitleText;
    engine::ui::Bindable<std::string> neuronLegendText;
    engine::ui::Bindable<std::string> neuronBiasText;
    engine::ui::Bindable<std::string> neuronNowText;
    engine::ui::Bindable<std::string> neuronInfluenceText;
    // False for a neuron that is a digit itself: it has no effect on the digits to chart.
    engine::ui::Bindable<bool> hasInfluence{false};
    engine::ui::Bindable<bool> weightsActive{false};
    engine::ui::Bindable<bool> weightsIdle{true};
    engine::ui::Bindable<bool> dreamActive{false};
    engine::ui::Bindable<bool> dreamIdle{true};
    engine::ui::MethodCommand neuronsLayerUp;
    engine::ui::MethodCommand neuronsLayerDown;
    engine::ui::MethodCommand showWeights;
    engine::ui::MethodCommand showDream;
    engine::ui::BindableList<std::shared_ptr<NeuronRow>> neuronRows;
    PicturePaint neuronPaint;
    PicturePaint drawingPaint;
    InfluencePaint influencePaint;

    // Train tab.
    engine::ui::Bindable<std::string> datasetNameText{"Dataset: MNIST"};
    engine::ui::Bindable<std::string> epochsText{"3"};
    engine::ui::Bindable<std::string> layersText{"1"};
    engine::ui::Bindable<std::string> neuronsText{"64"};
    engine::ui::Bindable<std::string> trainButtonText{"Train New"};
    engine::ui::Bindable<bool> shiftAugment{true};
    engine::ui::Bindable<bool> scaleAugment{false};
    engine::ui::Bindable<std::string> lastDatasetText{"No training yet"};
    engine::ui::Bindable<std::string> lastTrainText;
    engine::ui::Bindable<std::string> lastLossText;
    engine::ui::Bindable<std::string> lastSavedText;
    engine::ui::BindableList<std::shared_ptr<ListItem>> models;
    engine::ui::MethodCommand datasetPrev;
    engine::ui::MethodCommand datasetNext;
    engine::ui::MethodCommand epochsUp;
    engine::ui::MethodCommand epochsDown;
    engine::ui::MethodCommand layersUp;
    engine::ui::MethodCommand layersDown;
    engine::ui::MethodCommand neuronsUp;
    engine::ui::MethodCommand neuronsDown;
    engine::ui::MethodCommand train;
    engine::ui::MethodCommand continueTraining;
    engine::ui::MethodCommand loadModel;
    engine::ui::MethodCommand deleteModel;
    engine::ui::MethodCommand refreshModels;

    // Datasets tab.
    engine::ui::Bindable<std::string> datasetTitleText{"MNIST"};
    engine::ui::Bindable<std::string> sampleInfoText{"No sample selected"};
    engine::ui::Bindable<std::string> datasetCountsText{"Samples per digit: -"};
    engine::ui::Bindable<std::string> pageText{"Samples 0-0 of 0"};
    engine::ui::Bindable<std::string> newDatasetName;
    engine::ui::BindableList<std::shared_ptr<ListItem>> datasets;
    engine::ui::BindableList<std::shared_ptr<GridRow>> gridRows;
    engine::ui::MethodCommand pagePrev;
    engine::ui::MethodCommand pageNext;
    engine::ui::MethodCommand createDataset;
    engine::ui::MethodCommand deleteDataset;
    engine::ui::MethodCommand deleteSample;

    // Help tab. Topic i is a one-item list so that its document is built from the template; helpShown[i] says whether
    // it is the one on screen.
    engine::ui::BindableList<std::shared_ptr<HelpNavItem>> helpNav;
    engine::ui::Bindable<float> helpScroll{0.0f};
    std::array<engine::ui::BindableList<std::shared_ptr<HelpTopicViewModel>>, kHelpTopicCount> helpTopic;
    std::array<engine::ui::Bindable<bool>, kHelpTopicCount> helpShown;

private:
    enum class Tab { Recognize, Neurons, Train, Datasets, Help };

    void show_recognize() { show_tab(Tab::Recognize); }
    void show_neurons() { show_tab(Tab::Neurons); }
    void show_train() { show_tab(Tab::Train); }
    void show_datasets() { show_tab(Tab::Datasets); }
    void show_help() { show_tab(Tab::Help); }
    void show_tab(Tab tab);
};

}
