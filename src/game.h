#pragma once

#include "help/help.h"
#include "loc/tr.h"
#include "recognizer.h"
#include "view/recognizer_view_model.h"

#include <engine/core/engine_services.h>
#include <engine/core/job_system.h>
#include <engine/resources/assets_db.h>
#include <engine/igame.h>
#include <engine/ui/canvas.h>

#include <glm/geometric.hpp>
#include <glm/vec2.hpp>

#include <filesystem>
#include <memory>
#include <optional>
#include <string_view>

namespace game {

// One window with the recognizer screen. The drawing area is an IPaint on the screen's canvas; pointer events over it
// are polled from the UI pointer because a plain element has no pointer position to bind.
class Game final : public engine::GameBase {
public:
    explicit Game(const engine::EngineServices& services);

    [[nodiscard]] engine::WindowDesc primary_window() const override;
    [[nodiscard]] std::optional<engine::AssetId> window_icon() const override;
    void on_start() override;

private:
    void on_frame(engine::ecs::World& world);
    // Loads the English and Ukrainian messages and picks the language the player used last time.
    void load_locale();
    void set_locale(std::string_view tag);
    void use_english() { set_locale("en"); }
    void use_ukrainian() { set_locale("uk"); }
    // The pointer in the screen's design units, or nullopt before the canvas has a rect.
    [[nodiscard]] std::optional<glm::vec2> to_design(engine::ecs::World& world, glm::vec2 window_position) const;

    engine::JobSystem* jobs_;
    engine::AssetsDb* assets_;
    std::filesystem::path preferences_directory_;
    std::optional<Tr> tr_;
    std::shared_ptr<RecognizerViewModel> view_;
    std::optional<Recognizer> recognizer_;
    std::optional<Help> help_;
    engine::ecs::Entity screen_{};
    bool drawing_ = false;
    std::optional<glm::vec2> last_point_;  // normalized canvas position of the previous frame's stamp
};

}
