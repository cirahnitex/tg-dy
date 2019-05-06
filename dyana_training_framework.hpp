//
// Created by YAN Yuchen on 6/30/2018.
//

#ifndef DYANA_TRAINING_FRAMEWORK_HPP
#define DYANA_TRAINING_FRAMEWORK_HPP

#include <cmdlp.hpp>
#include <functional>
#include <dynet/dynet.h>
#include <dynet/io.h>
#include <json/json.h>
#include "dyana.hpp"
#include <ctime>
#include <exception_util.hpp>
#include <word2vec.hpp>
#include <cereal/archives/binary.hpp>
#include <cereal/types/vector.hpp>


namespace dyana {

  std::unordered_map<std::string, std::vector<float>> import_word2vec(const std::string &path) {
    std::unordered_map<std::string, std::vector<float> > embeddings;
    read_word2vec()(true,
                    util::id_func(),
                    util::curry(util::forced_map_lookup(), embeddings),
                    path);
    return embeddings;
  }

  /**
   * ensure that a directory exists, creating directories when necessary.
   * \param path
   */
  void ensure_dir(const std::string &path) {
    if (path.empty()) return;

    // recursively ensure parent directory
    std::size_t last_slash_pos = path.find_last_of('/');
    if (last_slash_pos != std::string::npos) ensure_dir(path.substr(0, last_slash_pos));

    // mkdir if doesn't exist
    struct stat st = {0};
    if (stat(path.c_str(), &st) == -1) {
      mkdir(path.c_str(), 0775);
    }
  }

  /**
   * get the filename of the latest file in a directory, according to the last modified time.
   * \param dir_path path to the directory
   * \param regex_filter a filter to match filename
   * \return the filename of the latest file, or empty of no matching files found
   */
  std::string get_latest_filename_in_directory(const std::string &dir_path, const std::string &regex_filter = "") {
    DIR *dp;
    struct dirent *dirp;
    if ((dp = opendir(dir_path.c_str())) == nullptr) {
      throw new std::runtime_error("Error( " + std::to_string(errno) + " ) opening " + dir_path);
    }

    time_t latest_modified_time = 0;
    std::string ret;
    while ((dirp = readdir(dp)) != nullptr) {
      std::string filename = std::string(dirp->d_name);

      // ignore the . and .. directory
      if (filename == "." || filename == "..") continue;

      // skip files that doesn't match the filter
      if (!regex_filter.empty() && !ECMAScript_string_utils::match(filename, regex_filter)) continue;

      // skip non-files
      struct stat file_stat;
      std::string file_path = dir_path + "/" + filename;
      if (stat(file_path.c_str(), &file_stat) == 0 && (file_stat.st_mode & S_IFREG)) {

        // store the latest filename
        auto modified_time_of_current_file = file_stat.st_mtim.tv_sec;
        if (modified_time_of_current_file > latest_modified_time) {
          latest_modified_time = modified_time_of_current_file;
          ret = filename;
        }
      }
    }
    closedir(dp);
    return ret;
  }

  class training_framework {
  public:

    struct training_options {
      std::string model_path;
      unsigned num_epochs;
      unsigned num_workers;

      void init(cmdlp::parser &p) {
        p.add(cmdlp::value_option<std::string>(model_path))
          ->name_value("file path")
          ->desc("the path where the model is stored")
          ->name("model_path");
        p.add(cmdlp::value_option<unsigned>(num_epochs))
          ->fallback(64)
          ->name_value("number")
          ->desc("number of epoches to run")
          ->name("num_epochs");
        p.add(cmdlp::value_option<unsigned>(num_workers))
          ->fallback(15)
          ->name_value("number")
          ->desc("the number of parallel processes to spawn")
          ->name("num_workers");
      }
    };

  };
}


#endif //DYANA_TRAINING_FRAMEWORK_HPP
