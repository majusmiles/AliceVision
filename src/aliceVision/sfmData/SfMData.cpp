// This file is part of the AliceVision project.
// Copyright (c) 2016 AliceVision contributors.
// Copyright (c) 2012 openMVG contributors.
// This Source Code Form is subject to the terms of the Mozilla Public License,
// v. 2.0. If a copy of the MPL was not distributed with this file,
// You can obtain one at https://mozilla.org/MPL/2.0/.

#include "SfMData.hpp"

#include <aliceVision/system/Logger.hpp>

#include <boost/filesystem.hpp>

namespace aliceVision {
namespace sfmData {

using namespace aliceVision::geometry;
using namespace aliceVision::camera;
using namespace aliceVision::image;

namespace fs = boost::filesystem;

SfMData::SfMData()
{
}

SfMData::~SfMData()
{
}

bool SfMData::operator==(const SfMData& other) const
{
    // Views
    if (views.size() != other.views.size())
        return false;

    for (Views::const_iterator it = views.begin(); it != views.end(); ++it)
    {
        const View& view1 = *(it->second.get());
        const View& view2 = *(other.views.at(it->first).get());
        if (view1 != view2)
            return false;

        // Image paths
        if (view1.getImagePath() != view2.getImagePath())
            return false;
    }

    // Poses
    if ((_poses != other._poses))
        return false;

    // Rigs
    if (_rigs != other._rigs)
        return false;

    // Intrinsics
    if (intrinsics.size() != other.intrinsics.size())
        return false;

    Intrinsics::const_iterator it = intrinsics.begin();
    Intrinsics::const_iterator otherIt = other.intrinsics.begin();
    for (; it != intrinsics.end() && otherIt != other.intrinsics.end(); ++it, ++otherIt)
    {
        // Index
        if (it->first != otherIt->first)
            return false;

        // Intrinsic
        camera::IntrinsicBase& intrinsic1 = *(it->second.get());
        camera::IntrinsicBase& intrinsic2 = *(otherIt->second.get());
        if (intrinsic1 != intrinsic2)
            return false;
    }

    // Points IDs are not preserved
    if (structure.size() != other.structure.size())
        return false;

    Landmarks::const_iterator landMarkIt = structure.begin();
    Landmarks::const_iterator otherLandmarkIt = other.structure.begin();
    for (; landMarkIt != structure.end() && otherLandmarkIt != other.structure.end(); ++landMarkIt, ++otherLandmarkIt)
    {
        // Points IDs are not preserved
        // Landmark
        const Landmark& landmark1 = landMarkIt->second;
        const Landmark& landmark2 = otherLandmarkIt->second;
        if (landmark1 != landmark2)
            return false;
    }

    // Control points
    if (control_points != other.control_points)
        return false;

    if (constraints2d.size() != other.constraints2d.size())
        return false;

    Constraints2D::const_iterator constraint2dIt = constraints2d.begin();
    Constraints2D::const_iterator otherconstraint2dIt = other.constraints2d.begin();
    for (; constraint2dIt != constraints2d.end() && otherconstraint2dIt != other.constraints2d.end(); ++constraint2dIt, ++otherconstraint2dIt)
    {
        if (*constraint2dIt != *otherconstraint2dIt)
            return false;
    }

    // Root path can be reseted during exports
    return true;
}

/**
 * @brief Convert paths in \p folders to absolute paths using \p absolutePath parent folder as base.
 * @param[in] folders list of paths to convert
 * @param[in] absolutePath filepath which parent folder should be used as base for absolute path conversion
 * @return the list of converted absolute paths or input folder if absolutePath is empty
 */
std::vector<std::string> toAbsoluteFolders(const std::vector<std::string>& folders, const std::string& absolutePath)
{
    // If absolute path is not set, return input folders
    if (absolutePath.empty())
        return folders;
    // Else, convert relative paths to absolute paths
    std::vector<std::string> absolutePaths;
    absolutePaths.reserve(folders.size());
    for (const auto& folder: folders)
    {
        const fs::path f = fs::absolute(folder, fs::path(absolutePath).parent_path());
        if (fs::exists(f))
        {
            // fs::canonical can only be used if the path exists
            absolutePaths.push_back(fs::canonical(f).string());
        }
        else
        {
            absolutePaths.push_back(f.string());
        }
    }
    return absolutePaths;
}

/** 
 * @brief Add paths contained in \p folders to \p dst as relative paths to \p absolutePath.
 *        Paths already present in \p dst are omitted.
 * @param[in] dst list in which paths should be added
 * @param[in] folders paths to add to \p dst as relative folders
 * @param[in] absolutePath filepath which parent folder should be used as base for relative path conversions
 */
void addAsRelativeFolders(std::vector<std::string>& dst, const std::vector<std::string>& folders, const std::string& absolutePath)
{
    for (auto folderPath: folders)
    {
        // If absolutePath is set, convert to relative path
        if (!absolutePath.empty() && fs::path(folderPath).is_absolute())
        {
            folderPath = fs::relative(folderPath, fs::path(absolutePath).parent_path()).string();
        }
        // Add path only if not already in dst
        if (std::find(dst.begin(), dst.end(), folderPath) == dst.end())
        {
            dst.emplace_back(folderPath);
        }
    }
}

std::vector<std::string> SfMData::getFeaturesFolders() const
{
    return toAbsoluteFolders(_featuresFolders, _absolutePath);
}

std::vector<std::string> SfMData::getMatchesFolders() const
{
    return toAbsoluteFolders(_matchesFolders, _absolutePath);
}

void SfMData::addFeaturesFolders(const std::vector<std::string>& folders)
{
    addAsRelativeFolders(_featuresFolders, folders, _absolutePath);
}

void SfMData::addMatchesFolders(const std::vector<std::string>& folders)
{
    addAsRelativeFolders(_matchesFolders, folders, _absolutePath);
}

void SfMData::setAbsolutePath(const std::string& path)
{
    // Get absolute path to features/matches folders
    const std::vector<std::string> featuresFolders = getFeaturesFolders();
    const std::vector<std::string> matchesFolders = getMatchesFolders();
    // Change internal absolute path
    _absolutePath = path;
    // Re-set features/matches folders
    // They will be converted back to relative paths based on updated _absolutePath
    setFeaturesFolders(featuresFolders);
    setMatchesFolders(matchesFolders);
}

std::set<IndexT> SfMData::getValidViews() const
{
    std::set<IndexT> valid_idx;
    for (Views::const_iterator it = views.begin(); it != views.end(); ++it)
    {
        const View * v = it->second.get();
        if (isPoseAndIntrinsicDefined(v))
        {
            valid_idx.insert(v->getViewId());
        }
    }
    return valid_idx;
}

std::set<IndexT> SfMData::getReconstructedIntrinsics() const
{
    std::set<IndexT> valid_idx;
    for (Views::const_iterator it = views.begin(); it != views.end(); ++it)
    {
        const View * v = it->second.get();
        if (isPoseAndIntrinsicDefined(v))
        {
            valid_idx.insert(v->getIntrinsicId());
        }
    }
    return valid_idx;
}

void SfMData::setPose(const View& view, const CameraPose& absolutePose)
{
    // const bool knownPose = existsPose(view);
    CameraPose& viewPose = _poses[view.getPoseId()];

    // Pose dedicated for this view (independant from rig, even if it is potentially part of a rig)
    if (view.isPoseIndependant())
    {
        viewPose.setTransform(absolutePose.getTransform());
        return;
    }

    // Initialized rig
    if (view.getRigId() != UndefinedIndexT)
    {
        const Rig& rig = _rigs.at(view.getRigId());
        RigSubPose& subPose = getRigSubPose(view);

        viewPose.setTransform(subPose.pose.inverse() * absolutePose.getTransform());
        return;
    }

    throw std::runtime_error("SfMData::setPose: dependant view pose not part of an initialized rig.");
}


void SfMData::combine(const SfMData& sfmData)
{
    if (!_rigs.empty() && !sfmData._rigs.empty())
        throw std::runtime_error("Can't combine two SfMData with rigs");

    // feature folder
    addFeaturesFolders(sfmData.getFeaturesFolders());

    // matching folder
    addMatchesFolders(sfmData.getMatchesFolders());

    // views
    views.insert(sfmData.views.begin(), sfmData.views.end());

    // intrinsics
    intrinsics.insert(sfmData.intrinsics.begin(), sfmData.intrinsics.end());

    // poses
    _poses.insert(sfmData._poses.begin(), sfmData._poses.end());

    // rigs
    _rigs.insert(sfmData._rigs.begin(), sfmData._rigs.end());

    // structure
    structure.insert(sfmData.structure.begin(), sfmData.structure.end());

    // control points
    control_points.insert(sfmData.control_points.begin(), sfmData.control_points.end());

    // constraints
    constraints2d.insert(constraints2d.end(), sfmData.constraints2d.begin(), sfmData.constraints2d.end());
}

void SfMData::clear()
{
    views.clear();
    intrinsics.clear();
    structure.clear();
    control_points.clear();
    _posesUncertainty.clear();
    _landmarksUncertainty.clear();
    constraints2d.clear();
    rotationpriors.clear();

    _absolutePath.clear();
    _featuresFolders.clear();
    _matchesFolders.clear();
    _poses.clear();
    _rigs.clear();
}

LandmarksPerView getLandmarksPerViews(const SfMData& sfmData)
{
    LandmarksPerView landmarksPerView;
    for (const auto& landIt : sfmData.getLandmarks())
    {
        for (const auto& obsIt : landIt.second.observations)
        {
            IndexT viewId = obsIt.first;
            LandmarkIdSet& landmarksSet = landmarksPerView[viewId];
            landmarksSet.push_back(landIt.first);
        }
    }

    // Sort landmark Ids in each view
    #pragma omp parallel for
    for (int i = 0; i < landmarksPerView.size(); ++i)
    {
        LandmarksPerView::iterator it = landmarksPerView.begin();
        std::advance(it, i);
        std::sort(it->second.begin(), it->second.end());
    }

    return landmarksPerView;
}

} // namespace sfmData
} // namespace aliceVision
