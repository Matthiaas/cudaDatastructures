#include "arg_parser.h"

#include <algorithm>
#include <charconv>

// This is taken from here:
// https://stackoverflow.com/questions/865668/parsing-command-line-arguments-in-c
arg_parser::arg_parser (int &argc, char **argv){
  for (int i=1; i < argc; ++i)
    this->tokens.push_back(std::string(argv[i]));
}

std::optional<const std::string_view> arg_parser::getCmdOption(const std::string &option) const {
  std::vector<std::string>::const_iterator itr;
  itr =  std::find(this->tokens.begin(), this->tokens.end(), option);
  if (itr != this->tokens.end() && ++itr != this->tokens.end()){
    return *itr;
  }
  return std::nullopt;
}

std::optional<size_t> arg_parser::getCmdOptionAsInt(const std::string &option) const {
  auto opt = getCmdOption(option);
  if(opt){
    size_t  res;
    auto exc = std::from_chars(opt->data(), opt->data() + opt->size(), res);
    if (exc.ec == std::errc::invalid_argument) {
      throw std::invalid_argument{"invalid_argument when parsing argument as size_t"};
    }
    else if (exc.ec == std::errc::result_out_of_range) {
      throw std::out_of_range{"out_of_range when parsing argument as size_t"};
    }
    return res;
  }
  return std::nullopt;
}


bool arg_parser::cmdOptionExists(const std::string &option) const{
  return std::find(this->tokens.begin(), this->tokens.end(), option)
         != this->tokens.end();
}
